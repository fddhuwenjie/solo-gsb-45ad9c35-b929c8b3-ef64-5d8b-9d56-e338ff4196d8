"""联单换发 API 测试。

覆盖：
- 案件更正后依据已有联单换发新联单：新单沿用原单批次顺序与 handoff_at，
  替换批次用新案件、未替换批次照旧，总体 released/held 按新单案件状态计算；
- 同一原单只能换发一次；相同请求幂等回放（含 replacements 顺序无关）；
- 争用同一原单的其他请求 409；新单号占用 409；新单可继续换发（换发链）；
- 原单以外的批次、替换成相同案件（无实际变更）409；
- 事务内逐批次重算核对：替换案件属于其他批次/已被替代/过期/未覆盖全部异常，
  以及未替换案件在换发时已失效，均 409 且不写入新单或换发关系；
- 原联单内容与同号回放保持历史快照；新单与普通建单同构，支持同内容回放；
- 结构校验（空 replacements、批次重复）422；原单不存在 404。
"""

import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="coldchain-reissue-"), "reissue.db")
os.environ["COLDCHAIN_DB"] = _TMP_DB

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


def iso(epoch_seconds: int) -> str:
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def make_payload(batch_id, device_id="reefer-01"):
    """两条异常：0–240（装货段）、1200–1260（运输段）。"""
    return {
        "batch_id": batch_id,
        "device_id": device_id,
        "gap_seconds": 600,
        "dip_tolerance_seconds": 120,
        "samples": [
            {"timestamp": iso(0), "temperature": 9.0},
            {"timestamp": iso(60), "temperature": 9.5},
            {"timestamp": iso(120), "temperature": 5.0},
            {"timestamp": iso(180), "temperature": 10.0},
            {"timestamp": iso(240), "temperature": 11.0},
            {"timestamp": iso(1200), "temperature": 9.0},
            {"timestamp": iso(1260), "temperature": 9.0},
        ],
        "segments": [
            {"segment_id": "seg-1", "segment_type": "loading",
             "start": iso(0), "end": iso(600),
             "lower_limit": 0.0, "upper_limit": 8.0},
            {"segment_id": "seg-2", "segment_type": "transport",
             "start": iso(600), "end": iso(1800),
             "lower_limit": 0.0, "upper_limit": 8.0},
        ],
    }


def make_clean_payload(batch_id):
    """无异常批次。"""
    return {
        "batch_id": batch_id,
        "device_id": "reefer-01",
        "samples": [
            {"timestamp": iso(0), "temperature": 5.0},
            {"timestamp": iso(60), "temperature": 6.0},
        ],
        "segments": [
            {"segment_id": "seg-1", "segment_type": "loading",
             "start": iso(0), "end": iso(600),
             "lower_limit": 0.0, "upper_limit": 8.0},
        ],
    }


def decisions_closed(note_a="现场已复冷", note_b="传感器故障，误报"):
    return [
        {"seq": 0, "decision": "resolved",
         "handled_at": iso(300), "note": note_a},
        {"seq": 1, "decision": "dismissed",
         "handled_at": iso(1300), "note": note_b},
    ]


def decisions_open():
    return [
        {"seq": 0, "decision": "accepted", "handled_at": iso(300)},
        {"seq": 1, "decision": "resolved",
         "handled_at": iso(1300), "note": "已处理"},
    ]


class ShipmentReissueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 与其他测试模块同进程运行时库路径可能已被固定，显式指向本模块独立库。
        db._DB_PATH = _TMP_DB
        db.init_db()
        cls.client = TestClient(app)

    # ---------- 工具 ----------

    @classmethod
    def _analyze(cls, payload):
        r = cls.client.post("/api/v1/batches/analyze", json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    @classmethod
    def _review(cls, case_id, batch_id, digest, decisions):
        r = cls.client.post("/api/v1/cases/review", json={
            "batch_id": batch_id,
            "case_id": case_id,
            "analysis_digest": digest,
            "decisions": decisions,
        })
        assert r.status_code == 200, r.text
        return r.json()

    @classmethod
    def _correct(cls, original, new, digest, decisions=None):
        r = cls.client.post("/api/v1/cases/correct", json={
            "original_case_id": original,
            "new_case_id": new,
            "analysis_digest": digest,
            "decisions": decisions if decisions is not None
            else decisions_closed("复核确认，更正后已复冷", "传感器故障，误报"),
        })
        assert r.status_code == 200, r.text
        return r.json()

    @classmethod
    def _make_case(cls, tag, decisions=None, clean=False):
        """建批次 + 处置，返回 (batch_id, case_id, digest)。"""
        batch_id = f"RB-{tag}"
        case_id = f"CASE-{tag}"
        result = cls._analyze(
            make_clean_payload(batch_id) if clean else make_payload(batch_id)
        )
        digest = result["analysis_digest"]
        dec = [] if clean else (
            decisions if decisions is not None else decisions_closed()
        )
        cls._review(case_id, batch_id, digest, dec)
        return batch_id, case_id, digest

    @staticmethod
    def _item(batch_id, digest, case_id, handoff_at):
        return {
            "batch_id": batch_id,
            "analysis_digest": digest,
            "case_id": case_id,
            "handoff_at": handoff_at,
        }

    def _ship(self, shipment_id, item_specs):
        """item_specs: [(batch_id, digest, case_id, handoff_epoch)]"""
        return self.client.post("/api/v1/shipments/snapshot", json={
            "shipment_id": shipment_id,
            "batches": [
                self._item(b, d, c, iso(t)) for b, d, c, t in item_specs
            ],
        })

    @staticmethod
    def _reissue_body(original, new, replacements):
        return {
            "original_shipment_id": original,
            "new_shipment_id": new,
            "replacements": [
                {"batch_id": b, "case_id": c} for b, c in replacements
            ],
        }

    def _reissue(self, original, new, replacements):
        return self.client.post(
            "/api/v1/shipments/reissue",
            json=self._reissue_body(original, new, replacements),
        )

    def _counts(self):
        with sqlite3.connect(_TMP_DB) as conn:
            shipments = conn.execute(
                "SELECT COUNT(*) FROM shipments").fetchone()[0]
            rows = conn.execute(
                "SELECT COUNT(*) FROM shipment_batches").fetchone()[0]
            reissues = conn.execute(
                "SELECT COUNT(*) FROM shipment_reissues").fetchone()[0]
        return shipments, rows, reissues

    def _make_two_batch_shipment(self, tag):
        """两张批次（A closed、B open）组成的 held 联单。"""
        ba, ca, da = self._make_case(f"{tag}-A")
        bb, cb, db_ = self._make_case(f"{tag}-B", decisions=decisions_open())
        r = self._ship(f"SHP-{tag}", [
            (ba, da, ca, 10000),
            (bb, db_, cb, 20000),
        ])
        assert r.status_code == 200, r.text
        return (ba, ca, da), (bb, cb, db_)

    # ---------- 用例 ----------

    def test_01_reissue_replaces_case_and_flips_result(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("01")
        # 更正 open 案件 CASE-01-B → 新案件 CASE-01-BR（closed）
        self._correct(cb, "CASE-01-BR", db_)

        r = self._reissue("SHP-01", "SHP-01R", [(bb, "CASE-01-BR")])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "MISS")
        body = r.json()
        self.assertFalse(body["cached"])
        self.assertEqual(body["original_shipment_id"], "SHP-01")
        self.assertEqual(body["shipment_id"], "SHP-01R")
        # 新单全部 closed → released
        self.assertEqual(body["result"], "released")
        self.assertEqual([b["seq"] for b in body["batches"]], [0, 1])
        # 沿用原单批次顺序
        self.assertEqual(
            [b["batch_id"] for b in body["batches"]], [ba, bb]
        )
        # 沿用原单 handoff_at
        self.assertEqual(
            [b["handoff_at"] for b in body["batches"]],
            [iso(10000).replace("Z", ".000Z"),
             iso(20000).replace("Z", ".000Z")],
        )
        # 未替换批次照旧；替换批次用新案件
        self.assertEqual(body["batches"][0]["case_id"], ca)
        self.assertEqual(body["batches"][1]["case_id"], "CASE-01-BR")
        self.assertEqual(
            [b["case_status"] for b in body["batches"]],
            ["closed", "closed"],
        )
        self.assertEqual(
            [b["anomaly_count"] for b in body["batches"]], [2, 2]
        )
        # 摘要为重算值
        self.assertEqual(body["batches"][0]["analysis_digest"], da)
        self.assertEqual(body["batches"][1]["analysis_digest"], db_)

        # 新联单与换发关系落库
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT result FROM shipments WHERE shipment_id = 'SHP-01R'"
                ).fetchone()[0],
                "released",
            )
            rows = conn.execute(
                """SELECT seq, batch_id, case_id FROM shipment_batches
                   WHERE shipment_id = 'SHP-01R' ORDER BY seq"""
            ).fetchall()
            self.assertEqual(
                [tuple(x) for x in rows],
                [(0, ba, ca), (1, bb, "CASE-01-BR")],
            )
            row = conn.execute(
                """SELECT new_shipment_id FROM shipment_reissues
                   WHERE original_shipment_id = 'SHP-01'"""
            ).fetchone()
            self.assertEqual(row[0], "SHP-01R")

    def test_02_held_result_when_replacement_case_open(self):
        # closed 案件被更正为 open：新单 held
        ba, ca, da = self._make_case("02-A")
        bb, cb, db_ = self._make_case("02-B")
        r = self._ship("SHP-02", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")

        self._correct(cb, "CASE-02-BR", db_, decisions=decisions_open())
        r = self._reissue("SHP-02", "SHP-02R", [(bb, "CASE-02-BR")])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "held")
        self.assertEqual(
            [b["case_status"] for b in r.json()["batches"]],
            ["closed", "open"],
        )

    def test_03_multiple_replacements_and_kept_batches(self):
        ba, ca, da = self._make_case("03-A")
        bb, cb, db_ = self._make_case("03-B")
        bc, cc, dc = self._make_case("03-C", clean=True)
        r = self._ship("SHP-03", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000), (bc, dc, cc, 30000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self._correct(ca, "CASE-03-AR", da)
        self._correct(cc, "CASE-03-CR", dc, decisions=[])

        r = self._reissue("SHP-03", "SHP-03R",
                          [(ba, "CASE-03-AR"), (bc, "CASE-03-CR")])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")
        self.assertEqual(
            [b["case_id"] for b in r.json()["batches"]],
            ["CASE-03-AR", cb, "CASE-03-CR"],
        )
        self.assertEqual(
            [b["anomaly_count"] for b in r.json()["batches"]], [2, 2, 0]
        )

    def test_04_same_reissue_replays_idempotently(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("04")
        self._correct(cb, "CASE-04-BR", db_)
        body = self._reissue_body("SHP-04", "SHP-04R",
                                  [(bb, "CASE-04-BR")])

        r1 = self.client.post("/api/v1/shipments/reissue", json=body)
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r1.headers.get("X-Cache"), "MISS")
        counts = self._counts()

        r2 = self.client.post("/api/v1/shipments/reissue", json=body)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertTrue(r2.json()["cached"])
        self.assertEqual(self._counts(), counts)

        # replacements 顺序不同仍视为同一换发请求
        multi_body = self._reissue_body("SHP-04", "SHP-04R",
                                        [(bb, "CASE-04-BR")])
        r3 = self.client.post("/api/v1/shipments/reissue", json=multi_body)
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")
        self.assertEqual(self._counts(), counts)

        # 除 cached 标记外回放内容完全一致
        j1, j2 = r1.json(), r2.json()
        self.assertFalse(j1.pop("cached"))
        self.assertTrue(j2.pop("cached"))
        self.assertEqual(j1, j2)

    def test_05_replacements_order_insensitive_with_two_items(self):
        ba, ca, da = self._make_case("05-A")
        bb, cb, db_ = self._make_case("05-B")
        self._ship("SHP-05", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self._correct(ca, "CASE-05-AR", da)
        self._correct(cb, "CASE-05-BR", db_)
        body1 = self._reissue_body("SHP-05", "SHP-05R",
                                   [(ba, "CASE-05-AR"),
                                    (bb, "CASE-05-BR")])
        r1 = self.client.post("/api/v1/shipments/reissue", json=body1)
        self.assertEqual(r1.status_code, 200, r1.text)
        counts = self._counts()

        body2 = self._reissue_body("SHP-05", "SHP-05R",
                                   [(bb, "CASE-05-BR"),
                                    (ba, "CASE-05-AR")])
        r2 = self.client.post("/api/v1/shipments/reissue", json=body2)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertEqual(self._counts(), counts)

    def test_06_original_can_only_be_reissued_once(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("06")
        self._correct(cb, "CASE-06-BR", db_)
        self.assertEqual(
            self._reissue("SHP-06", "SHP-06R", [(bb, "CASE-06-BR")]).status_code,
            200,
        )
        counts = self._counts()

        # 同一原单换新单号再换发 → 409
        r = self._reissue("SHP-06", "SHP-06R2", [(bb, "CASE-06-BR")])
        self.assertEqual(r.status_code, 409)
        self.assertIn("SHP-06R", r.json()["detail"])
        # 同一原单同一新单号但换替换案件 → 409（内容不同）
        self._correct(ca, "CASE-06-AR", da)
        r = self._reissue("SHP-06", "SHP-06R", [(ba, "CASE-06-AR")])
        self.assertEqual(r.status_code, 409)

        # 拒绝不留下部分记录
        self.assertEqual(self._counts(), counts)
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM shipments WHERE shipment_id = 'SHP-06R2'"
                ).fetchone()
            )

    def test_07_unknown_original_shipment_404(self):
        ba, ca, da = self._make_case("07")
        r = self._reissue("SHP-NOPE", "SHP-07R", [(ba, ca)])
        self.assertEqual(r.status_code, 404)
        self.assertIn("SHP-NOPE", r.json()["detail"])

    def test_08_new_shipment_id_must_be_unique(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("08")
        # 另一张已存在的普通联单占用新单号
        self.assertEqual(
            self._ship("SHP-08-OTHER", [(ba, da, ca, 5000)]).status_code,
            200,
        )
        counts = self._counts()

        # 新单号被普通联单占用 → 409
        r = self._reissue("SHP-08", "SHP-08-OTHER", [(bb, cb)])
        self.assertEqual(r.status_code, 409)
        self.assertIn("SHP-08-OTHER", r.json()["detail"])
        # 新单号与原单号相同 → 409（原单号占用）
        r = self._reissue("SHP-08", "SHP-08", [(bb, cb)])
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)

    def test_09_batch_not_in_original_rejected(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("09")
        bx, cx, dx = self._make_case("09-X")
        counts = self._counts()

        # 原单以外的批次 → 409
        r = self._reissue("SHP-09", "SHP-09R", [(bx, cx)])
        self.assertEqual(r.status_code, 409)
        self.assertIn(bx, r.json()["detail"])
        # 夹带原单外批次与合法替换 → 仍整体 409
        self._correct(cb, "CASE-09-BR", db_)
        r = self._reissue("SHP-09", "SHP-09R",
                          [(bb, "CASE-09-BR"), (bx, cx)])
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)

    def test_10_replacement_must_actually_change_case(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("10")
        counts = self._counts()

        # 替换案件与原案件相同 → 409（无实际变更）
        r = self._reissue("SHP-10", "SHP-10R", [(bb, cb)])
        self.assertEqual(r.status_code, 409)
        self.assertIn(bb, r.json()["detail"])
        self.assertIn(cb, r.json()["detail"])
        # 全部无变更 → 409（且模型层不允许空 replacements）
        r = self._reissue("SHP-10", "SHP-10R", [(ba, ca)])
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)

    def test_11_empty_or_duplicate_replacements_422(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("11")
        self._correct(cb, "CASE-11-BR", db_)
        counts = self._counts()

        # 空 replacements → 422
        r = self.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": "SHP-11",
            "new_shipment_id": "SHP-11R",
            "replacements": [],
        })
        self.assertEqual(r.status_code, 422)
        # 同批次重复 → 422
        r = self.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": "SHP-11",
            "new_shipment_id": "SHP-11R",
            "replacements": [
                {"batch_id": bb, "case_id": "CASE-11-BR"},
                {"batch_id": bb, "case_id": "CASE-11-BR"},
            ],
        })
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self._counts(), counts)

    def test_12_replacement_case_of_other_batch_rejected(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("12")
        counts = self._counts()
        # CASE-12-A 属于 ba，却用来替换 bb
        r = self._reissue("SHP-12", "SHP-12R", [(bb, ca)])
        self.assertEqual(r.status_code, 409)
        self.assertIn(ca, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_13_replaced_case_rejected_for_reissue(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("13")
        # A 的原案件 ca 被更正为 CASE-13-AR；换发时仍拿已被替代的 ca
        # 去替换 A 批次 → 409（与普通建单同一规则）
        self._correct(ca, "CASE-13-AR", da)
        counts = self._counts()
        r = self._reissue("SHP-13", "SHP-13R", [(ba, ca)])
        self.assertEqual(r.status_code, 409)
        self.assertIn(ca, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_14_replacement_nonexistent_case_rejected(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("14")
        counts = self._counts()
        r = self._reissue("SHP-14", "SHP-14R",
                          [(bb, "CASE-14-NOPE")])
        self.assertEqual(r.status_code, 409)
        self.assertIn("CASE-14-NOPE", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_15_kept_case_also_revalidated_and_rejects_when_dead(self):
        """未替换的案件在换发时同样被核对：已被替代 → 409，新单不落库。"""
        ba, ca, da = self._make_case("15-A")
        bb, cb, db_ = self._make_case("15-B", decisions=decisions_open())
        r = self._ship("SHP-15", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        # B 批次（替换项）更正为 closed 新案件
        self._correct(cb, "CASE-15-BR", db_)
        # A 批次未列入替换，但其原案件 ca 也已被更正 → 换发必须失败
        self._correct(ca, "CASE-15-AR", da)
        counts = self._counts()

        r = self._reissue("SHP-15", "SHP-15R", [(bb, "CASE-15-BR")])
        self.assertEqual(r.status_code, 409)
        self.assertIn(ca, r.json()["detail"])
        self.assertEqual(self._counts(), counts)
        # 补上 A 的替换后换发成功
        r = self._reissue(
            "SHP-15", "SHP-15R",
            [(bb, "CASE-15-BR"), (ba, "CASE-15-AR")],
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")

    def test_16_original_snapshot_and_same_id_replay_preserved(self):
        ba, ca, da = self._make_case("16-A")
        bb, cb, db_ = self._make_case("16-B", decisions=decisions_open())
        r1 = self._ship("SHP-16", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r1.json()["result"], "held")
        self._correct(cb, "CASE-16-BR", db_)

        rr = self._reissue("SHP-16", "SHP-16R", [(bb, "CASE-16-BR")])
        self.assertEqual(rr.status_code, 200, rr.text)

        # 原联单同号回放保持历史快照：仍 held、仍引用旧案件 cb
        r2 = self._ship("SHP-16", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertEqual(r2.json()["result"], "held")
        self.assertEqual(
            [b["case_id"] for b in r2.json()["batches"]], [ca, cb]
        )
        j1, j2 = r1.json(), r2.json()
        self.assertFalse(j1.pop("cached"))
        self.assertTrue(j2.pop("cached"))
        self.assertEqual(j1, j2)

    def test_17_new_shipment_behaves_like_normal_waybill(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("17")
        self._correct(cb, "CASE-17-BR", db_)
        rr = self._reissue("SHP-17", "SHP-17R", [(bb, "CASE-17-BR")])
        self.assertEqual(rr.status_code, 200, rr.text)

        # 新单可按普通建单接口同内容幂等回放（content_hash 口径一致）
        replay = {
            "shipment_id": "SHP-17R",
            "batches": [
                self._item(ba, da, ca, iso(10000)),
                self._item(bb, db_, "CASE-17-BR", iso(20000)),
            ],
        }
        r = self.client.post("/api/v1/shipments/snapshot", json=replay)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")
        body = r.json()
        self.assertTrue(body["cached"])
        self.assertEqual(body["result"], "released")
        self.assertEqual(
            [b["case_id"] for b in body["batches"]], [ca, "CASE-17-BR"]
        )
        # 换内容仍 409（新单不可变更）
        changed = json.loads(json.dumps(replay))
        changed["batches"][1]["handoff_at"] = iso(20001)
        r = self.client.post("/api/v1/shipments/snapshot", json=changed)
        self.assertEqual(r.status_code, 409)

    def test_18_new_shipment_can_be_reissued_again(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("18")
        self._correct(cb, "CASE-18-BR", db_)
        r1 = self._reissue("SHP-18", "SHP-18R", [(bb, "CASE-18-BR")])
        self.assertEqual(r1.status_code, 200, r1.text)

        # 对新单继续更正、继续换发（换发链）
        self._correct("CASE-18-BR", "CASE-18-BR2", db_)
        r2 = self._reissue(
            "SHP-18R", "SHP-18R2", [(bb, "CASE-18-BR2")]
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.json()["original_shipment_id"], "SHP-18R")
        self.assertEqual(r2.json()["shipment_id"], "SHP-18R2")
        self.assertEqual(
            [b["case_id"] for b in r2.json()["batches"]],
            [ca, "CASE-18-BR2"],
        )

        # 第一级换发的同内容回放仍可用
        r3 = self._reissue("SHP-18", "SHP-18R", [(bb, "CASE-18-BR")])
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")

    def test_19_replacement_case_on_stale_snapshot_rejected(self):
        """替换案件依据的摘要与当前重算不一致 → 409（直接播种过期案件）。"""
        import time

        ba, ca, da = self._make_case("19-A")
        bb, cb, db_ = self._make_case("19-B", decisions=decisions_open())
        r = self._ship("SHP-19", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        # 在 bb 上播种一个绑定错误摘要的“新案件”，伪装成更正后的案件
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                """INSERT INTO cases
                   (case_id, batch_id, analysis_digest, content_hash,
                    status, created_at)
                   VALUES (?, ?, ?, ?, 'closed', ?)""",
                ("CASE-19-STALE", bb, "f" * 64, "seed", time.time()),
            )
            conn.executemany(
                """INSERT INTO case_decisions
                   (case_id, seq, decision, handled_at, note)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    ("CASE-19-STALE", 0, "resolved", 300, None),
                    ("CASE-19-STALE", 1, "resolved", 1300, None),
                ],
            )
        counts = self._counts()
        r = self._reissue("SHP-19", "SHP-19R",
                          [(bb, "CASE-19-STALE")])
        self.assertEqual(r.status_code, 409)
        self.assertIn("CASE-19-STALE", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_20_replacement_case_not_covering_anomalies_rejected(self):
        """替换案件未覆盖全部当前异常 → 409（播种残缺案件）。"""
        import time

        ba, ca, da = self._make_case("20-A")
        bb, cb, db_ = self._make_case("20-B", decisions=decisions_open())
        r = self._ship("SHP-20", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                """INSERT INTO cases
                   (case_id, batch_id, analysis_digest, content_hash,
                    status, created_at)
                   VALUES (?, ?, ?, ?, 'closed', ?)""",
                ("CASE-20-PART", bb, db_, "seed", time.time()),
            )
            conn.execute(
                """INSERT INTO case_decisions
                   (case_id, seq, decision, handled_at, note)
                   VALUES ('CASE-20-PART', 0, 'resolved', 300, NULL)""",
            )
        counts = self._counts()
        r = self._reissue("SHP-20", "SHP-20R",
                          [(bb, "CASE-20-PART")])
        self.assertEqual(r.status_code, 409)
        self.assertIn("CASE-20-PART", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_21_concurrent_reissues_only_one_wins(self):
        (ba, ca, da), (bb, cb, db_) = self._make_two_batch_shipment("21")
        self._correct(cb, "CASE-21-BR", db_)
        # 两组不同请求争用同一原单
        body_x = self._reissue_body("SHP-21", "SHP-21X",
                                    [(bb, "CASE-21-BR")])
        body_y = self._reissue_body("SHP-21", "SHP-21Y",
                                    [(bb, "CASE-21-BR")])
        bodies = [body_x, body_y] * 4

        def submit(body):
            return self.client.post("/api/v1/shipments/reissue", json=body)

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(submit, bodies))

        ok = [r for r in responses if r.status_code == 200]
        conflicts = [r for r in responses if r.status_code == 409]
        self.assertEqual(len(ok) + len(conflicts), len(bodies))
        self.assertGreaterEqual(len(ok), 1)
        self.assertGreaterEqual(len(conflicts), 1)
        winners = {r.json()["shipment_id"] for r in ok}
        self.assertEqual(len(winners), 1)
        winner = winners.pop()
        loser = "SHP-21Y" if winner == "SHP-21X" else "SHP-21X"

        # 争用失败的请求不落新单、不产生第二条换发关系
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM shipments WHERE shipment_id = ?", (loser,)
                ).fetchone()
            )
            self.assertEqual(
                conn.execute(
                    """SELECT COUNT(*) FROM shipment_reissues
                       WHERE original_shipment_id = 'SHP-21'"""
                ).fetchone()[0],
                1,
            )
        # 胜出的换发可幂等回放
        winning_body = body_x if winner == "SHP-21X" else body_y
        r = self.client.post("/api/v1/shipments/reissue", json=winning_body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")

    def test_22_digest_and_anomaly_count_recomputed_per_batch(self):
        """新单的 digest/异常数从事务内重算结果取，不照抄原单行。"""
        ba, ca, da = self._make_case("22-A")
        bb, cb, db_ = self._make_case("22-B")
        r = self._ship("SHP-22", [
            (ba, da, ca, 10000), (bb, db_, cb, 20000),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self._correct(cb, "CASE-22-BR", db_)
        # 把原单行的 digest/异常数改坏；换发仍应取重算值而非照抄
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                """UPDATE shipment_batches SET analysis_digest = ?,
                   anomaly_count = 99 WHERE shipment_id = 'SHP-22'""",
                ("deadbeef",),
            )
        rr = self._reissue("SHP-22", "SHP-22R", [(bb, "CASE-22-BR")])
        self.assertEqual(rr.status_code, 200, rr.text)
        self.assertEqual(
            [b["analysis_digest"] for b in rr.json()["batches"]],
            [da, db_],
        )
        self.assertEqual(
            [b["anomaly_count"] for b in rr.json()["batches"]], [2, 2]
        )


if __name__ == "__main__":
    unittest.main()
