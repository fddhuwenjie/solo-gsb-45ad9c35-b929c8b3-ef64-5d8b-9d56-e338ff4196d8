"""处置结论更正 API 测试。

覆盖：
- 更正在同一事务内写入新案件与替代关系，响应返回新案件状态与受影响联单；
- 相同原案件、新案件号与处置内容重复提交幂等回放（含 decisions 顺序无关）；
- 摘要过期 / 原案件已被替代 / 新案件号被占用 → 409 且不留下部分记录；
- 逐条处置校验规则与复核一致（覆盖、序号、说明、时间）；
- 被替代案件不得用于新联单；已有联单内容及同号回放保持历史快照；
- 原复核接口仍可回放旧案件；新案件可开新联单、可继续更正（更正链）；
- 并发争用同一原案件的更正只有一个成功。
"""

import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="coldchain-correct-"), "correct.db")
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


def decisions_closed():
    """原处置：seq 0 已解决、seq 1 驳回（误报）。"""
    return [
        {"seq": 0, "decision": "resolved",
         "handled_at": iso(300), "note": "现场已复冷"},
        {"seq": 1, "decision": "dismissed",
         "handled_at": iso(1300), "note": "传感器故障，误报"},
    ]


def decisions_corrected():
    """更正后的结论：seq 0 由 resolved 改为 dismissed（复核确认误报）。"""
    return [
        {"seq": 0, "decision": "dismissed",
         "handled_at": iso(400), "note": "复核确认为传感器误报，更正原结论"},
        {"seq": 1, "decision": "dismissed",
         "handled_at": iso(1300), "note": "传感器故障，误报"},
    ]


class CaseCorrectionTests(unittest.TestCase):
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
    def _make_case(cls, tag, decisions=None):
        """建一个批次并完成一次处置复核，返回 (batch_id, case_id, digest)。"""
        batch_id = f"CB-{tag}"
        case_id = f"CASE-{tag}"
        digest = cls._analyze(make_payload(batch_id))
        cls._review(case_id, batch_id, digest,
                    decisions if decisions is not None else decisions_closed())
        return batch_id, case_id, digest

    @staticmethod
    def _correct_body(original, new, digest, decisions=None):
        return {
            "original_case_id": original,
            "new_case_id": new,
            "analysis_digest": digest,
            "decisions": decisions if decisions is not None
            else decisions_corrected(),
        }

    def _counts(self):
        with sqlite3.connect(_TMP_DB) as conn:
            cases = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            corrections = conn.execute(
                "SELECT COUNT(*) FROM case_corrections"
            ).fetchone()[0]
        return cases, corrections

    def _ship(self, shipment_id, batch_id, digest, case_id):
        return self.client.post("/api/v1/shipments/snapshot", json={
            "shipment_id": shipment_id,
            "batches": [{"batch_id": batch_id, "analysis_digest": digest,
                         "case_id": case_id, "handoff_at": iso(10000)}],
        })

    # ---------- 用例 ----------

    def test_01_correction_creates_new_case_and_relation(self):
        batch_id, case_id, digest = self._make_case("01")
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-01R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "MISS")
        body = r.json()
        self.assertFalse(body["cached"])
        self.assertEqual(body["batch_id"], batch_id)
        self.assertEqual(body["original_case_id"], case_id)
        self.assertEqual(body["case_id"], "CASE-01R")
        self.assertEqual(body["analysis_digest"], digest)
        self.assertEqual(body["status"], "closed")
        self.assertEqual([d["seq"] for d in body["decisions"]], [0, 1])
        self.assertEqual(body["decisions"][0]["decision"], "dismissed")
        self.assertEqual(body["decisions"][0]["handled_at"],
                         iso(400).replace("Z", ".000Z"))
        self.assertEqual(body["affected_shipment_ids"], [])

        # 替代关系与新案件落库，原案件保持不变
        with sqlite3.connect(_TMP_DB) as conn:
            row = conn.execute(
                """SELECT new_case_id FROM case_corrections
                   WHERE original_case_id = ?""",
                (case_id,),
            ).fetchone()
            self.assertEqual(row[0], "CASE-01R")
            self.assertEqual(
                conn.execute(
                    "SELECT status FROM cases WHERE case_id = 'CASE-01R'"
                ).fetchone()[0],
                "closed",
            )
            self.assertEqual(
                conn.execute(
                    """SELECT COUNT(*) FROM case_decisions
                       WHERE case_id = 'CASE-01R'"""
                ).fetchone()[0],
                2,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT status FROM cases WHERE case_id = ?",
                    (case_id,),
                ).fetchone()[0],
                "closed",
            )

    def test_02_new_case_status_follows_review_rules(self):
        _, case_id, digest = self._make_case("02")
        decisions = [
            {"seq": 0, "decision": "accepted", "handled_at": iso(300)},
            {"seq": 1, "decision": "resolved",
             "handled_at": iso(1300), "note": "已处理"},
        ]
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-02R", digest, decisions),
        )
        self.assertEqual(r.status_code, 200, r.text)
        # 存在 accepted → 新案件 open
        self.assertEqual(r.json()["status"], "open")
        self.assertIsNone(r.json()["decisions"][0]["note"])

    def test_03_same_correction_replays_idempotently(self):
        _, case_id, digest = self._make_case("03")
        body = self._correct_body(case_id, "CASE-03R", digest)
        r1 = self.client.post("/api/v1/cases/correct", json=body)
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r1.headers.get("X-Cache"), "MISS")
        counts = self._counts()

        r2 = self.client.post("/api/v1/cases/correct", json=body)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertTrue(r2.json()["cached"])
        # 幂等回放不产生新案件、新替代关系
        self.assertEqual(self._counts(), counts)

        # decisions 顺序不同仍视为同一更正（规范化哈希按 seq 排序）
        reordered = dict(body)
        reordered["decisions"] = list(reversed(body["decisions"]))
        r3 = self.client.post("/api/v1/cases/correct", json=reordered)
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")
        self.assertEqual(self._counts(), counts)

        # 除 cached 标记外回放内容完全一致
        j1, j2 = r1.json(), r2.json()
        self.assertFalse(j1.pop("cached"))
        self.assertTrue(j2.pop("cached"))
        self.assertEqual(j1, j2)

    def test_04_original_case_can_only_be_corrected_once(self):
        _, case_id, digest = self._make_case("04")
        body = self._correct_body(case_id, "CASE-04R", digest)
        self.assertEqual(
            self.client.post("/api/v1/cases/correct", json=body).status_code,
            200,
        )
        counts = self._counts()

        # 同一原案件换个新案件号再更正 → 409
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-04R2", digest),
        )
        self.assertEqual(r.status_code, 409)
        self.assertIn("CASE-04R", r.json()["detail"])
        # 同一原案件同一新案件号但改结论 → 409（内容不同）
        changed = self._correct_body(case_id, "CASE-04R", digest)
        changed["decisions"][0]["note"] = "另一种说法"
        r = self.client.post("/api/v1/cases/correct", json=changed)
        self.assertEqual(r.status_code, 409)
        # 同一原案件换摘要 → 409（内容不同）
        changed = self._correct_body(case_id, "CASE-04R", digest)
        changed["analysis_digest"] = "0" * 64
        r = self.client.post("/api/v1/cases/correct", json=changed)
        self.assertEqual(r.status_code, 409)

        # 拒绝不留下部分记录：没有新案件、没有第二条替代关系
        self.assertEqual(self._counts(), counts)
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM cases WHERE case_id = 'CASE-04R2'"
                ).fetchone()
            )

    def test_05_unknown_original_case_404(self):
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body("CASE-NOPE", "CASE-05R", "0" * 64),
        )
        self.assertEqual(r.status_code, 404)

    def test_06_stale_digest_conflicts_and_persists_nothing(self):
        _, case_id, digest = self._make_case("06")
        counts = self._counts()
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-06R", "0" * 64),
        )
        self.assertEqual(r.status_code, 409)
        # 过期拒绝：没有新案件、没有替代关系
        self.assertEqual(self._counts(), counts)
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM cases WHERE case_id = 'CASE-06R'"
                ).fetchone()
            )
        # 拒绝不留副作用：用正确摘要仍可更正
        r2 = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-06R", digest),
        )
        self.assertEqual(r2.status_code, 200, r2.text)

    def test_07_new_case_id_must_be_unique(self):
        _, case_id, digest = self._make_case("07")
        _, other_case, _ = self._make_case("07B")
        counts = self._counts()

        # 新案件号与既有案件重号 → 409
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, other_case, digest),
        )
        self.assertEqual(r.status_code, 409)
        # 新案件号与原案件号相同 → 409（原案件占用了该号）
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, case_id, digest),
        )
        self.assertEqual(r.status_code, 409)
        # 占用拒绝不留下部分记录
        self.assertEqual(self._counts(), counts)

    def test_08_decisions_validation_matches_review_rules(self):
        _, case_id, digest = self._make_case("08")
        base = decisions_corrected()
        counts = self._counts()

        # 缺序号 1
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-08A", digest, [base[0]]),
        )
        self.assertEqual(r.status_code, 422)
        # 越界序号 2
        extra = base + [
            {"seq": 2, "decision": "resolved", "handled_at": iso(9999)}
        ]
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-08B", digest, extra),
        )
        self.assertEqual(r.status_code, 422)
        # dismissed 缺说明
        no_note = [dict(base[0]), dict(base[1])]
        no_note[0].pop("note")
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-08C", digest, no_note),
        )
        self.assertEqual(r.status_code, 422)
        # handled_at 早于异常结束（seq 0 结束于 240）
        early = [dict(base[0], handled_at=iso(239)), dict(base[1])]
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-08D", digest, early),
        )
        self.assertEqual(r.status_code, 422)
        # 重复序号（模型层拒绝）
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(
                case_id, "CASE-08E", digest, [base[0], base[0]]
            ),
        )
        self.assertEqual(r.status_code, 422)

        # 全部 422 拒绝后仍可用合法内容更正（校验失败不落库）
        self.assertEqual(self._counts(), counts)
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-08R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)

    def test_09_new_case_can_be_corrected_again(self):
        _, case_id, digest = self._make_case("09")
        r1 = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-09R", digest),
        )
        self.assertEqual(r1.status_code, 200, r1.text)

        # 更正生成的新案件可继续更正（更正链）
        r2 = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body("CASE-09R", "CASE-09R2", digest),
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.json()["original_case_id"], "CASE-09R")
        self.assertEqual(r2.json()["case_id"], "CASE-09R2")
        self.assertEqual(r2.json()["batch_id"], r1.json()["batch_id"])

        # 第一级更正的同内容回放仍可用
        r3 = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-09R", digest),
        )
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")

    def test_10_replaced_case_rejected_for_new_shipment(self):
        batch_id, case_id, digest = self._make_case("10")
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-10R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)

        # 被替代案件开新联单 → 409，且联单不落库
        r = self._ship("SHP-10-OLD", batch_id, digest, case_id)
        self.assertEqual(r.status_code, 409)
        self.assertIn(case_id, r.json()["detail"])
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM shipments WHERE shipment_id = 'SHP-10-OLD'"
                ).fetchone()
            )

        # 更正后的新案件可正常开联单
        r = self._ship("SHP-10-NEW", batch_id, digest, "CASE-10R")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["result"], "released")
        self.assertEqual(r.json()["batches"][0]["case_id"], "CASE-10R")
        self.assertEqual(r.json()["batches"][0]["case_status"], "closed")

    def test_11_existing_shipment_snapshot_survives_correction(self):
        batch_id, case_id, digest = self._make_case("11")
        r1 = self._ship("SHP-11", batch_id, digest, case_id)
        self.assertEqual(r1.status_code, 200, r1.text)

        # 更正后响应列出直接引用原案件的联单
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-11R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["affected_shipment_ids"], ["SHP-11"])

        # 已有联单同号回放保持历史快照（仍引用原案件）
        r2 = self._ship("SHP-11", batch_id, digest, case_id)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertEqual(r2.json()["batches"][0]["case_id"], case_id)
        j1, j2 = r1.json(), r2.json()
        self.assertFalse(j1.pop("cached"))
        self.assertTrue(j2.pop("cached"))
        self.assertEqual(j1, j2)

        # 更正的幂等回放同样返回受影响联单
        r3 = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-11R", digest),
        )
        self.assertEqual(r3.headers.get("X-Cache"), "HIT")
        self.assertEqual(r3.json()["affected_shipment_ids"], ["SHP-11"])

    def test_12_affected_shipments_sorted_by_shipment_id(self):
        batch_id, case_id, digest = self._make_case("12")
        # 故意按非字典序创建三张引用同一案件的联单
        for ship_id in ("SHP-12-Z", "SHP-12-A", "SHP-12-M"):
            r = self._ship(ship_id, batch_id, digest, case_id)
            self.assertEqual(r.status_code, 200, r.text)

        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-12R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            r.json()["affected_shipment_ids"],
            ["SHP-12-A", "SHP-12-M", "SHP-12-Z"],
        )

    def test_13_review_endpoint_still_replays_replaced_case(self):
        batch_id, case_id, digest = self._make_case("13")
        review_body = {
            "batch_id": batch_id,
            "case_id": case_id,
            "analysis_digest": digest,
            "decisions": decisions_closed(),
        }
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-13R", digest),
        )
        self.assertEqual(r.status_code, 200, r.text)

        # 原复核接口仍可回放旧案件（历史快照不变）
        r2 = self.client.post("/api/v1/cases/review", json=review_body)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertTrue(r2.json()["cached"])
        self.assertEqual(r2.json()["status"], "closed")
        self.assertEqual(
            [d["decision"] for d in r2.json()["decisions"]],
            ["resolved", "dismissed"],
        )
        # 旧案件依旧不可变更：换内容 409
        changed = json.loads(json.dumps(review_body))
        changed["decisions"][0]["note"] = "篡改"
        r3 = self.client.post("/api/v1/cases/review", json=changed)
        self.assertEqual(r3.status_code, 409)

    def test_14_concurrent_corrections_only_one_wins(self):
        _, case_id, digest = self._make_case("14")
        body_a = self._correct_body(case_id, "CASE-14A", digest)
        body_b = self._correct_body(case_id, "CASE-14B", digest)
        bodies = [body_a, body_b] * 4

        def submit(body):
            return self.client.post("/api/v1/cases/correct", json=body)

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(submit, bodies))

        ok = [r for r in responses if r.status_code == 200]
        conflicts = [r for r in responses if r.status_code == 409]
        # 每个请求都有确定结果；成功的都指向同一个新案件号
        self.assertEqual(len(ok) + len(conflicts), len(bodies))
        self.assertGreaterEqual(len(ok), 1)
        self.assertGreaterEqual(len(conflicts), 1)
        winners = {r.json()["case_id"] for r in ok}
        self.assertEqual(len(winners), 1)
        winner = winners.pop()
        loser = "CASE-14B" if winner == "CASE-14A" else "CASE-14A"

        # 争用同一原案件的其他更正全部失败，且未留下部分记录
        with sqlite3.connect(_TMP_DB) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM cases WHERE case_id = ?", (loser,)
                ).fetchone()
            )
            self.assertEqual(
                conn.execute(
                    """SELECT COUNT(*) FROM case_corrections
                       WHERE original_case_id = ?""",
                    (case_id,),
                ).fetchone()[0],
                1,
            )
        # 胜出的更正可幂等回放
        winning_body = body_a if winner == "CASE-14A" else body_b
        r = self.client.post("/api/v1/cases/correct", json=winning_body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")

    def test_15_zero_anomaly_case_correction(self):
        digest = self._analyze({
            "batch_id": "CB-15",
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
        })
        self._review("CASE-15", "CB-15", digest, [])
        r = self.client.post("/api/v1/cases/correct", json={
            "original_case_id": "CASE-15",
            "new_case_id": "CASE-15R",
            "analysis_digest": digest,
            "decisions": [],
        })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "closed")
        self.assertEqual(r.json()["decisions"], [])

    def test_16_digest_of_other_batch_conflicts(self):
        _, case_id, digest = self._make_case("16")
        # 另一个异常形态不同的批次 → 摘要不同
        other = make_payload("CB-16B")
        other["samples"][0]["temperature"] = 5.0
        other_digest = self._analyze(other)
        self.assertNotEqual(digest, other_digest)
        counts = self._counts()
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-16R", other_digest),
        )
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self._counts(), counts)

    def test_17_new_case_behaves_like_reviewed_case(self):
        batch_id, case_id, digest = self._make_case("17")
        decisions = decisions_corrected()
        r = self.client.post(
            "/api/v1/cases/correct",
            json=self._correct_body(case_id, "CASE-17R", digest, decisions),
        )
        self.assertEqual(r.status_code, 200, r.text)

        # 新案件可用复核接口按同内容幂等回放
        r2 = self.client.post("/api/v1/cases/review", json={
            "batch_id": batch_id,
            "case_id": "CASE-17R",
            "analysis_digest": digest,
            "decisions": decisions,
        })
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.headers.get("X-Cache"), "HIT")
        self.assertEqual(r2.json()["status"], "closed")
        # 换内容仍 409（新案件同样不可变更）
        changed = {
            "batch_id": batch_id,
            "case_id": "CASE-17R",
            "analysis_digest": digest,
            "decisions": [dict(decisions[0], note="篡改"), decisions[1]],
        }
        r3 = self.client.post("/api/v1/cases/review", json=changed)
        self.assertEqual(r3.status_code, 409)


if __name__ == "__main__":
    unittest.main()
