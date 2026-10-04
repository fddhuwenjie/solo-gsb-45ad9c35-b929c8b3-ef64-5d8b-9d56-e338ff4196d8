"""运输联单换发 API 测试。

覆盖：
- 换发沿用原单批次顺序与 handoff_at，替换批次用新案件、其余照旧；
- 新单 released/held 按新单案件状态计算（原 held 可转 released，反之亦然）；
- 同一事务内逐批重算 analysis_digest，案件属于对应批次、覆盖当前异常、
  未被替代，任一项失效（含未替换批次上的旧案件已被替代）整单拒绝、不留行；
- 原单以外的批次、无实际变更、新单号占用（含与原单同号）→ 409；
- 同一原单只能换发一次；相同请求回放幂等（替换项顺序无关）；争用 409；
- 原联单内容与同号回放保持历史快照；新单可继续换发（换发链）、可在快照
  接口按同内容回放；原分析/复核/更正/建单接口保持可用；
- 空替换/重复批次 → 422；原单不存在 → 404。
"""

import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="coldchain-reissue-"), "reissue.db")
os.environ["COLDCHAIN_DB"] = _TMP_DB

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import _reissue_content_hash, app  # noqa: E402
from app.models import ShipmentReissueRequest  # noqa: E402

client: TestClient


def iso(epoch_seconds: int) -> str:
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def make_two_anomaly_payload(batch_id):
    """两条异常：0–240（装货段）、1200–1260（运输段）。"""
    return {
        "batch_id": batch_id,
        "device_id": "reefer-01",
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


def make_normal_payload(batch_id):
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


def decisions_resolved():
    return [
        {"seq": 0, "decision": "resolved",
         "handled_at": iso(300), "note": "已复冷"},
        {"seq": 1, "decision": "resolved",
         "handled_at": iso(1300), "note": "已处理"},
    ]


def decisions_accepted():
    return [
        {"seq": 0, "decision": "accepted", "handled_at": iso(300)},
        {"seq": 1, "decision": "resolved",
         "handled_at": iso(1300), "note": "已处理"},
    ]


def decisions_dismissed():
    return [
        {"seq": 0, "decision": "dismissed",
         "handled_at": iso(400), "note": "复核确认为传感器误报"},
        {"seq": 1, "decision": "dismissed",
         "handled_at": iso(1300), "note": "传感器故障，误报"},
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
        return r.json()["analysis_digest"]

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
    def _correct(cls, old_case, new_case, digest, decisions):
        r = cls.client.post("/api/v1/cases/correct", json={
            "original_case_id": old_case,
            "new_case_id": new_case,
            "analysis_digest": digest,
            "decisions": decisions,
        })
        assert r.status_code == 200, r.text
        return r.json()

    @classmethod
    def _ship(cls, shipment_id, items):
        return cls.client.post("/api/v1/shipments/snapshot", json={
            "shipment_id": shipment_id,
            "batches": items,
        })

    @classmethod
    def _reissue(cls, original, new, replacements):
        return cls.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": original,
            "new_shipment_id": new,
            "replacements": replacements,
        })

    @staticmethod
    def _item(batch_id, digest, case_id, handoff_at):
        return {
            "batch_id": batch_id,
            "analysis_digest": digest,
            "case_id": case_id,
            "handoff_at": handoff_at,
        }

    def _batch(self, tag, suffix, zero=False):
        batch_id = f"R-{tag}-{suffix}"
        payload = make_normal_payload(batch_id) if zero \
            else make_two_anomaly_payload(batch_id)
        return batch_id, self._analyze(payload)

    def _make_original(self, tag, specs, shipment_id=None,
                       start=10000, step=10000):
        """按 specs 创建批次、案件与原联单。

        specs 元素为 (suffix, decisions)；decisions 为 None/[] 表示无异常
        批次。返回 (shipment_id, info)，info 元素为
        (batch_id, digest, case_id, handoff_at)，顺序即联单顺序。
        """
        items, info = [], []
        for index, (suffix, decisions) in enumerate(specs):
            zero = not decisions
            batch_id, digest = self._batch(tag, suffix, zero=zero)
            case_id = f"CASE-{tag}-{suffix}"
            self._review(case_id, batch_id, digest,
                         [] if zero else decisions)
            handoff = iso(start + index * step)
            items.append(self._item(batch_id, digest, case_id, handoff))
            info.append((batch_id, digest, case_id, handoff))
        shipment_id = shipment_id or f"SHP-{tag}"
        r = self._ship(shipment_id, items)
        self.assertEqual(r.status_code, 200, r.text)
        return shipment_id, info

    def _counts(self):
        with sqlite3.connect(_TMP_DB) as conn:
            shipments = conn.execute(
                "SELECT COUNT(*) FROM shipments").fetchone()[0]
            rows = conn.execute(
                "SELECT COUNT(*) FROM shipment_batches").fetchone()[0]
            reissues = conn.execute(
                "SELECT COUNT(*) FROM shipment_reissues").fetchone()[0]
        return shipments, rows, reissues

    @staticmethod
    def _closed_specs(*suffixes):
        return [(s, decisions_resolved()) for s in suffixes]

    # ---------- 用例 ----------

    def test_01_reissue_replaces_case_keeps_order_and_handoff(self):
        tag = "01"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
            ("C", []),
        ])
        batch_a, digest_a, case_a, _ = info[0]
        _, digest_b, case_b, _ = info[1]
        batch_c, digest_c, case_c, _ = info[2]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        counts = self._counts()

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": batch_a, "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "MISS")
        body = r.json()
        self.assertFalse(body["cached"])
        self.assertEqual(body["original_shipment_id"], ship_id)
        self.assertEqual(body["shipment_id"], f"{ship_id}R")
        # CASE-AR closed，但 CASE-B 仍 open → held
        self.assertEqual(body["result"], "held")
        self.assertEqual(
            [(b["seq"], b["batch_id"], b["case_id"]) for b in body["batches"]],
            [
                (0, batch_a, case_ar),
                (1, "R-01-B", case_b),
                (2, batch_c, case_c),
            ],
        )
        # 批次顺序与 handoff_at 沿用原单
        self.assertEqual(
            [b["handoff_at"] for b in body["batches"]],
            [iso(10000).replace("Z", ".000Z"),
             iso(20000).replace("Z", ".000Z"),
             iso(30000).replace("Z", ".000Z")],
        )
        self.assertEqual(
            [b["analysis_digest"] for b in body["batches"]],
            [digest_a, digest_b, digest_c],
        )
        self.assertEqual(
            [b["anomaly_count"] for b in body["batches"]], [2, 2, 0]
        )
        self.assertEqual(
            [b["case_status"] for b in body["batches"]],
            ["closed", "open", "closed"],
        )
        # 落库：新联单 + 3 批次行 + 1 换发关系
        self.assertEqual(
            self._counts(), (counts[0] + 1, counts[1] + 3, counts[2] + 1)
        )
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT new_shipment_id FROM shipment_reissues "
                    "WHERE original_shipment_id = ?",
                    (ship_id,),
                ).fetchone()[0],
                f"{ship_id}R",
            )
            self.assertEqual(
                conn.execute(
                    "SELECT result FROM shipments WHERE shipment_id = ?",
                    (ship_id,),
                ).fetchone()[0],
                "held",
            )

    def test_02_reissue_can_turn_held_into_released(self):
        tag = "02"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        _, digest_b, case_b, _ = info[1]
        case_ar, case_br = f"CASE-{tag}-AR", f"CASE-{tag}-BR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        self._correct(case_b, case_br, digest_b, decisions_resolved())

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
            {"batch_id": f"R-{tag}-B", "case_id": case_br},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")
        self.assertEqual(
            [b["case_id"] for b in r.json()["batches"]],
            [case_ar, case_br, f"CASE-{tag}-C"],
        )
        self.assertEqual(
            [b["case_status"] for b in r.json()["batches"]],
            ["closed", "closed", "closed"],
        )

    def test_03_reissue_can_turn_released_into_held(self):
        tag = "03"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ao = f"CASE-{tag}-AO"
        self._correct(case_a, case_ao, digest_a, decisions_accepted())

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ao},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "held")
        self.assertEqual(r.json()["batches"][0]["case_status"], "open")
        self.assertEqual(r.json()["batches"][1]["case_id"], f"CASE-{tag}-C")

    def test_04_same_reissue_replays_idempotently(self):
        tag = "04"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        _, _, case_b, _ = info[1]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())

        body = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}R",
            "replacements": [{"batch_id": f"R-{tag}-A", "case_id": case_ar}],
        }
        r1 = self.client.post("/api/v1/shipments/reissue", json=body)
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r1.headers.get("X-Cache"), "MISS")
        counts = self._counts()

        r2 = self.client.post("/api/v1/shipments/reissue", json=body)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertTrue(r2.json()["cached"])
        self.assertEqual(self._counts(), counts)

        # 除 cached 外回放内容完全一致
        j1, j2 = r1.json(), r2.json()
        self.assertFalse(j1.pop("cached"))
        self.assertTrue(j2.pop("cached"))
        self.assertEqual(j1, j2)

        # replacements 顺序不影响规范化哈希
        body_xy = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}RX",
            "replacements": [
                {"batch_id": f"R-{tag}-A", "case_id": case_ar},
                {"batch_id": f"R-{tag}-B", "case_id": case_b},
            ],
        }
        body_yx = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}RX",
            "replacements": [
                {"batch_id": f"R-{tag}-B", "case_id": case_b},
                {"batch_id": f"R-{tag}-A", "case_id": case_ar},
            ],
        }
        self.assertEqual(
            _reissue_content_hash(ShipmentReissueRequest.model_validate(body_xy)),
            _reissue_content_hash(ShipmentReissueRequest.model_validate(body_yx)),
        )
        # 换新联单号/任一案件都会得到不同哈希
        body_other = jsonish_deep(body)
        body_other["new_shipment_id"] = f"{ship_id}RZ"
        self.assertNotEqual(
            _reissue_content_hash(ShipmentReissueRequest.model_validate(body)),
            _reissue_content_hash(
                ShipmentReissueRequest.model_validate(body_other)
            ),
        )

    def test_05_same_original_can_only_be_reissued_once(self):
        tag = "05"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        _, _, case_b, _ = info[1]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())

        body = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}R",
            "replacements": [{"batch_id": f"R-{tag}-A", "case_id": case_ar}],
        }
        self.assertEqual(
            self.client.post("/api/v1/shipments/reissue",
                             json=body).status_code,
            200,
        )
        counts = self._counts()

        # 换新联单号 → 409（详情指向已换发的新单）
        changed = jsonish_deep(body)
        changed["new_shipment_id"] = f"{ship_id}R2"
        r = self.client.post("/api/v1/shipments/reissue", json=changed)
        self.assertEqual(r.status_code, 409)
        self.assertIn(f"{ship_id}R", r.json()["detail"])
        # 增挂替换项 → 409（内容不同）
        changed = jsonish_deep(body)
        changed["replacements"].append(
            {"batch_id": f"R-{tag}-B", "case_id": case_b}
        )
        r = self.client.post("/api/v1/shipments/reissue", json=changed)
        self.assertEqual(r.status_code, 409)
        # 换替换案件 → 409
        changed = jsonish_deep(body)
        changed["replacements"][0]["case_id"] = case_a
        r = self.client.post("/api/v1/shipments/reissue", json=changed)
        self.assertEqual(r.status_code, 409)
        # 拒绝不留下部分记录
        self.assertEqual(self._counts(), counts)
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM shipments WHERE shipment_id = ?",
                    (f"{ship_id}R2",),
                ).fetchone()
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM shipment_reissues "
                    "WHERE original_shipment_id = ?",
                    (ship_id,),
                ).fetchone()[0],
                1,
            )

    def test_06_unknown_original_shipment_404(self):
        counts = self._counts()
        r = self._reissue("SHP-NOPE-06", "SHP-06R", [
            {"batch_id": "R-X", "case_id": "CASE-X"},
        ])
        self.assertEqual(r.status_code, 404)
        self.assertIn("SHP-NOPE-06", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_07_new_shipment_id_must_be_unique(self):
        tag = "07"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        # 另一张已存在的联单
        occupied = self._ship(f"SHP-{tag}OCC", [
            self._item(f"R-{tag}-A", digest_a, case_ar, iso(50000)),
        ])
        self.assertEqual(occupied.status_code, 200)
        counts = self._counts()

        # 新单号与既有联单重号 → 409
        r = self._reissue(ship_id, f"SHP-{tag}OCC", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(f"SHP-{tag}OCC", r.json()["detail"])
        # 新单号与原联单同号 → 409
        r = self._reissue(ship_id, ship_id, [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)

    def test_08_batch_not_in_original_rejected(self):
        tag = "08"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        # 已分析但不在原联单上的批次
        other_batch, other_digest = self._batch(tag, "X", zero=True)
        self._review(f"CASE-{tag}-X", other_batch, other_digest, [])
        counts = self._counts()

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": other_batch, "case_id": f"CASE-{tag}-X"},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(other_batch, r.json()["detail"])
        # 同时夹带一个合法替换也整体拒绝
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
            {"batch_id": other_batch, "case_id": f"CASE-{tag}-X"},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(other_batch, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_09_no_actual_change_rejected_but_mixed_allowed(self):
        tag = "09"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        counts = self._counts()

        # 替换项仍指向原案件（零项实际变更）→ 409
        r = self._reissue(ship_id, f"{ship_id}X", [
            {"batch_id": f"R-{tag}-B", "case_id": f"CASE-{tag}-B"},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)
        # 一个真变更 + 一个同号空变更：允许
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
            {"batch_id": f"R-{tag}-B", "case_id": f"CASE-{tag}-B"},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            [b["case_id"] for b in r.json()["batches"]],
            [case_ar, f"CASE-{tag}-B"],
        )
        self.assertEqual(
            self._counts(), (counts[0] + 1, counts[1] + 2, counts[2] + 1)
        )

    def test_10_duplicate_replacement_batch_422(self):
        counts = self._counts()
        r = self.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": "SHP-WHATEVER-10",
            "new_shipment_id": "SHP-10R",
            "replacements": [
                {"batch_id": "R-X", "case_id": "CASE-X1"},
                {"batch_id": "R-X", "case_id": "CASE-X2"},
            ],
        })
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self._counts(), counts)

    def test_11_empty_or_missing_replacements_422(self):
        counts = self._counts()
        r = self.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": "SHP-WHATEVER-11",
            "new_shipment_id": "SHP-11R",
            "replacements": [],
        })
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/api/v1/shipments/reissue", json={
            "original_shipment_id": "SHP-WHATEVER-11",
            "new_shipment_id": "SHP-11R",
        })
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self._counts(), counts)

    def test_12_new_case_must_belong_to_batch(self):
        tag = "12"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        counts = self._counts()

        # CASE-AR 属于 A，却拿去替换 B
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-B", "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(case_ar, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_13_unknown_new_case_rejected(self):
        tag = "13"
        ship_id, _ = self._make_original(tag, [
            ("A", decisions_resolved()),
        ])
        counts = self._counts()
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": f"CASE-{tag}-NOPE"},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(f"CASE-{tag}-NOPE", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_14_replaced_case_rejected_in_reissue(self):
        tag = "14"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        _, digest_c, case_c, _ = info[1]
        case_ar, case_cr = f"CASE-{tag}-AR", f"CASE-{tag}-CR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        self._correct(case_c, case_cr, digest_c, [])
        counts = self._counts()

        # A→旧案件（已被替代），C→新案件（真变更）：逐批核对先撞上 A 已替代
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_a},
            {"batch_id": f"R-{tag}-C", "case_id": case_cr},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(case_a, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_15_unreplaced_superseded_case_rejects_whole_reissue(self):
        tag = "15"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        _, digest_c, case_c, _ = info[1]
        case_ar, case_cr = f"CASE-{tag}-AR", f"CASE-{tag}-CR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        self._correct(case_c, case_cr, digest_c, [])
        counts = self._counts()

        # 只替换 C（真变更）；A 未替换且旧案件已被替代 → 整单 409
        r = self._reissue(ship_id, f"{ship_id}X", [
            {"batch_id": f"R-{tag}-C", "case_id": case_cr},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(case_a, r.json()["detail"])
        self.assertEqual(self._counts(), counts)

        # 正确做法：A、C 都改用更正后的案件
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
            {"batch_id": f"R-{tag}-C", "case_id": case_cr},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            [b["case_id"] for b in r.json()["batches"]],
            [case_ar, case_cr],
        )

    def test_16_unreplaced_stale_case_rejects_whole_reissue(self):
        tag = "16"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("C", []),
        ])
        batch_a, digest_a, case_a, _ = info[0]
        _, digest_c, case_c, _ = info[1]
        case_cr = f"CASE-{tag}-CR"
        self._correct(case_c, case_cr, digest_c, [])
        # 把 A 的案件摘要改坏，模拟处置依据过期（未替换批次同样要核对）
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                "UPDATE cases SET analysis_digest = ? WHERE case_id = ?",
                ("f" * 64, case_a),
            )
        counts = self._counts()

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-C", "case_id": case_cr},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(case_a, r.json()["detail"])
        self.assertEqual(self._counts(), counts)
        # 原联单同号回放仍是历史快照（换发整体失败不影响它）
        r2 = self._ship(ship_id, [
            self._item(batch_a, digest_a, case_a, iso(10000)),
            self._item(f"R-{tag}-C", digest_c, case_c, iso(20000)),
        ])
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")

    def test_17_new_case_not_covering_anomalies_rejected(self):
        tag = "17"
        batch_id, digest = self._batch(tag, "A")
        # 直接播种原联单引用残缺案件 CASE-X（只有 seq 0）
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                "INSERT INTO cases "
                "(case_id, batch_id, analysis_digest, content_hash, "
                " status, created_at) VALUES (?, ?, ?, ?, 'closed', 0)",
                (f"CASE-{tag}-X", batch_id, digest, "seed1"),
            )
            conn.execute(
                "INSERT INTO case_decisions "
                "(case_id, seq, decision, handled_at, note) "
                "VALUES (?, 0, 'resolved', 300, NULL)",
                (f"CASE-{tag}-X",),
            )
            conn.execute(
                "INSERT INTO shipments "
                "(shipment_id, content_hash, result, created_at) "
                "VALUES (?, 'seed', 'released', 0)",
                (f"SHP-{tag}",),
            )
            conn.execute(
                "INSERT INTO shipment_batches "
                "(shipment_id, seq, batch_id, case_id, analysis_digest, "
                " handoff_at, anomaly_count, case_status) "
                "VALUES (?, 0, ?, ?, ?, 10000, 2, 'closed')",
                (f"SHP-{tag}", batch_id, f"CASE-{tag}-X", digest),
            )
            # 替换目标：同批次另一个残缺案件（只有 seq 1）
            conn.execute(
                "INSERT INTO cases "
                "(case_id, batch_id, analysis_digest, content_hash, "
                " status, created_at) VALUES (?, ?, ?, ?, 'closed', 0)",
                (f"CASE-{tag}-Y", batch_id, digest, "seed2"),
            )
            conn.execute(
                "INSERT INTO case_decisions "
                "(case_id, seq, decision, handled_at, note) "
                "VALUES (?, 1, 'resolved', 1300, NULL)",
                (f"CASE-{tag}-Y",),
            )
        counts = self._counts()

        r = self._reissue(f"SHP-{tag}", f"SHP-{tag}R", [
            {"batch_id": batch_id, "case_id": f"CASE-{tag}-Y"},
        ])
        self.assertEqual(r.status_code, 409)
        self.assertIn(f"CASE-{tag}-Y", r.json()["detail"])
        self.assertEqual(self._counts(), counts)

    def test_18_original_shipment_remains_historical_snapshot(self):
        tag = "18"
        ship_id, original_info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
            ("C", []),
        ])
        original_body = {
            "shipment_id": ship_id,
            "batches": [
                self._item(b, d, c, h) for b, d, c, h in original_info
            ],
        }
        r1 = self._ship(ship_id, original_body["batches"])
        _, digest_a, case_a, _ = original_info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())

        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 200, r.text)

        # 原联单同号回放保持历史快照（仍引用旧案件，即使旧案件已被替代）
        r2 = self._ship(ship_id, original_body["batches"])
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertTrue(r2.json()["cached"])
        self.assertEqual(
            [b["case_id"] for b in r2.json()["batches"]],
            [c for _, _, c, _ in original_info],
        )
        self.assertEqual(r2.json()["result"], r1.json()["result"])

        # 库中两张联单并存，批次行各自独立
        with sqlite3.connect(_TMP_DB) as conn:
            rows = conn.execute(
                "SELECT shipment_id, batch_id, case_id, handoff_at "
                "FROM shipment_batches WHERE shipment_id IN (?, ?) "
                "ORDER BY shipment_id, seq",
                (ship_id, f"{ship_id}R"),
            ).fetchall()
        expected = []
        for b, _, c, h in original_info:
            expected.append((ship_id, b, c, _epoch(h)))
        expected.append((f"{ship_id}R", f"R-{tag}-A", case_ar, _epoch(iso(10000))))
        expected.append((f"{ship_id}R", f"R-{tag}-B", f"CASE-{tag}-B",
                         _epoch(iso(20000))))
        expected.append((f"{ship_id}R", f"R-{tag}-C", f"CASE-{tag}-C",
                         _epoch(iso(30000))))
        self.assertEqual([tuple(x) for x in rows], expected)

    def test_19_new_shipment_replays_on_snapshot_endpoint(self):
        tag = "19"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        new_batches = [
            self._item(b, d, c, h)
            for b, d, c, h in [
                (info[0][0], info[0][1], case_ar, info[0][3]),
                info[1],
            ]
        ]

        # 新联单是普通联单：快照接口按同内容幂等回放
        r2 = self._ship(f"{ship_id}R", new_batches)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertEqual(
            [b["case_id"] for b in r2.json()["batches"]],
            [case_ar, f"CASE-{tag}-B"],
        )
        # 新联单同样不可变：换内容 409
        changed = list(new_batches)
        changed[0] = self._item(f"R-{tag}-A", digest_a, case_ar, iso(10001))
        self.assertEqual(
            self._ship(f"{ship_id}R", changed).status_code, 409
        )

    def test_20_new_shipment_can_be_reissued_again_chain(self):
        tag = "20"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
            ("C", []),
        ])
        _, digest_a, case_a, _ = info[0]
        _, digest_b, case_b, _ = info[1]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())

        r1 = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r1.status_code, 200, r1.text)

        # 对换发出的新联单继续换发：B 改判 closed
        case_br = f"CASE-{tag}-BR"
        self._correct(case_b, case_br, digest_b, decisions_resolved())
        r2 = self._reissue(f"{ship_id}R", f"{ship_id}R2", [
            {"batch_id": f"R-{tag}-B", "case_id": case_br},
        ])
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.json()["original_shipment_id"], f"{ship_id}R")
        self.assertEqual(r2.json()["shipment_id"], f"{ship_id}R2")
        self.assertEqual(r2.json()["result"], "released")
        self.assertEqual(
            [b["case_id"] for b in r2.json()["batches"]],
            [case_ar, case_br, f"CASE-{tag}-C"],
        )
        # 交接时间仍沿用最初原单
        self.assertEqual(
            [b["handoff_at"] for b in r2.json()["batches"]],
            [iso(10000).replace("Z", ".000Z"),
             iso(20000).replace("Z", ".000Z"),
             iso(30000).replace("Z", ".000Z")],
        )
        # 两条换发关系并存
        with sqlite3.connect(_TMP_DB) as conn:
            chain = conn.execute(
                "SELECT original_shipment_id, new_shipment_id "
                "FROM shipment_reissues WHERE original_shipment_id IN (?, ?) "
                "ORDER BY original_shipment_id",
                (ship_id, f"{ship_id}R"),
            ).fetchall()
        self.assertEqual(
            [tuple(x) for x in chain],
            [(ship_id, f"{ship_id}R"), (f"{ship_id}R", f"{ship_id}R2")],
        )
        # 第一级换发的同内容回放仍可用
        r3 = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-A", "case_id": case_ar},
        ])
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")

    def test_21_concurrent_reissues_only_one_wins(self):
        tag = "21"
        ship_id, info = self._make_original(tag, [
            ("A", decisions_resolved()),
            ("B", decisions_accepted()),
        ])
        _, digest_a, case_a, _ = info[0]
        case_ar = f"CASE-{tag}-AR"
        self._correct(case_a, case_ar, digest_a, decisions_dismissed())

        body_a = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}A",
            "replacements": [{"batch_id": f"R-{tag}-A", "case_id": case_ar}],
        }
        body_b = {
            "original_shipment_id": ship_id,
            "new_shipment_id": f"{ship_id}B",
            "replacements": [
                {"batch_id": f"R-{tag}-A", "case_id": case_ar},
                {"batch_id": f"R-{tag}-B", "case_id": f"CASE-{tag}-B"},
            ],
        }
        bodies = [body_a, body_b] * 4
        counts_before = self._counts()

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
        loser = f"{ship_id}B" if winner == f"{ship_id}A" else f"{ship_id}A"

        # 争用同一原联单的其他换发全部失败，未留下部分记录
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM shipments WHERE shipment_id = ?", (loser,)
                ).fetchone()
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM shipment_reissues "
                    "WHERE original_shipment_id = ?",
                    (ship_id,),
                ).fetchone()[0],
                1,
            )
        shipments, batch_rows, reissues = self._counts()
        self.assertEqual(shipments, counts_before[0] + 1)
        self.assertEqual(batch_rows, counts_before[1] + 2)
        self.assertEqual(reissues, counts_before[2] + 1)
        # 胜出请求可幂等回放
        winning_body = body_a if winner == f"{ship_id}A" else body_b
        r = self.client.post(
            "/api/v1/shipments/reissue", json=winning_body
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")

    def test_22_other_endpoints_still_available(self):
        tag = "22"
        batch_id, digest = self._batch(tag, "A")
        self._review(f"CASE-{tag}-A", batch_id, digest, decisions_resolved())
        r = self._ship(f"SHP-{tag}", [
            self._item(batch_id, digest, f"CASE-{tag}-A", iso(70000)),
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")
        r = self.client.post("/api/v1/cases/correct", json={
            "original_case_id": f"CASE-{tag}-A",
            "new_case_id": f"CASE-{tag}-AR",
            "analysis_digest": digest,
            "decisions": decisions_dismissed(),
        })
        self.assertEqual(r.status_code, 200, r.text)
        # GET 回放分析仍可用
        r = self.client.get(f"/api/v1/batches/{batch_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["analysis_digest"], digest)
        # 原复核案件同内容回放仍可用
        r = self.client.post("/api/v1/cases/review", json={
            "batch_id": batch_id,
            "case_id": f"CASE-{tag}-A",
            "analysis_digest": digest,
            "decisions": decisions_resolved(),
        })
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")
        # 换发仍可用
        r = self._reissue(f"SHP-{tag}", f"SHP-{tag}R", [
            {"batch_id": batch_id, "case_id": f"CASE-{tag}-AR"},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["batches"][0]["case_id"], f"CASE-{tag}-AR")

    def test_23_single_batch_shipment_reissue(self):
        tag = "23"
        ship_id, info = self._make_original(tag, [("C", [])])
        _, digest_c, case_c, handoff = info[0]
        self._correct(case_c, f"CASE-{tag}-CR", digest_c, [])
        r = self._reissue(ship_id, f"{ship_id}R", [
            {"batch_id": f"R-{tag}-C", "case_id": f"CASE-{tag}-CR"},
        ])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")
        self.assertEqual(len(r.json()["batches"]), 1)
        self.assertEqual(r.json()["batches"][0]["case_id"], f"CASE-{tag}-CR")
        self.assertEqual(
            r.json()["batches"][0]["handoff_at"],
            handoff.replace("Z", ".000Z"),
        )


def jsonish_deep(value):
    import copy
    return copy.deepcopy(value)


def _epoch(iso_str):
    from datetime import datetime
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


if __name__ == "__main__":
    unittest.main()
