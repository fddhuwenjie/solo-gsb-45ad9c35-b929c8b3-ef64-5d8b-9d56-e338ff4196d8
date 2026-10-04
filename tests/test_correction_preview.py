"""校准档案拟更正预演测试。

覆盖：
- 同一读取快照内按原档案与拟定区间分别重算：两份摘要、异常数、变化标记；
- 仅摘要变化的批次列出绑定旧摘要的 case_id / shipment_id（排序去重）；
- 异常列表未变但档案内容改变（区间拆分）同样置变化标记并列出受影响对象；
- 绑定其他摘要（过期）的案件/联单不出现在受影响清单；
- 档案/批次不存在、设备/档案不匹配、拟定区间漏采样 → 指出对象并整次拒绝；
- 预演纯只读：档案、批次、处置、联单在预演前后逐字节一致；
- 请求结构校验：批次号重复、数量越界、区间非法。
"""

import json
import os
import sqlite3
import tempfile
import unittest

_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="coldchain-preview-"), "preview.db")
os.environ["COLDCHAIN_DB"] = _TMP_DB

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


def iso(epoch_seconds) -> str:
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def iso_out(epoch_seconds) -> str:
    return iso(epoch_seconds).replace("Z", ".000Z")


DEVICE = "reefer-dry-01"
PROFILE = "CAL-DRY"
PREVIEW_URL = "/api/v1/calibration-profiles/correction-preview"

TABLES = (
    "batches",
    "anomalies",
    "cases",
    "case_decisions",
    "shipments",
    "shipment_batches",
    "calibration_profiles",
    "calibration_intervals",
)


def analysis_body(batch_id, samples, device_id=DEVICE, profile_id=PROFILE):
    body = {
        "batch_id": batch_id,
        "device_id": device_id,
        "gap_seconds": 600,
        "dip_tolerance_seconds": 120,
        "samples": samples,
        "segments": [
            {"segment_id": "seg-1", "segment_type": "loading",
             "start": iso(0), "end": iso(600),
             "lower_limit": 0.0, "upper_limit": 8.0},
        ],
    }
    if profile_id is not None:
        body["profile_id"] = profile_id
    return body


def preview_body(batch_ids, intervals, profile_id=PROFILE):
    return {
        "profile_id": profile_id,
        "intervals": intervals,
        "batch_ids": batch_ids,
    }


class CorrectionPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db._DB_PATH = _TMP_DB
        db.init_db()
        cls.client = TestClient(app)

        # 原档案：单段 [0,1800) corrected = raw * 1.0 + 0.5
        r = cls.client.post(
            "/api/v1/calibration-profiles",
            json={
                "profile_id": PROFILE,
                "device_id": DEVICE,
                "intervals": [
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            },
        )
        assert r.status_code == 200, r.text
        # 另一设备的另一档案（用于"档案不匹配"用例）
        r = cls.client.post(
            "/api/v1/calibration-profiles",
            json={
                "profile_id": "CAL-OTHER",
                "device_id": DEVICE,
                "intervals": [
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.1},
                ],
            },
        )
        assert r.status_code == 200, r.text

        # PREV-B1：校准后 8.5 / 5.5(回落) / 9.5 → 1 条异常 [0,120]
        r = cls.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B1",
                samples=[
                    {"timestamp": iso(0), "temperature": 8.0},
                    {"timestamp": iso(60), "temperature": 5.0},
                    {"timestamp": iso(120), "temperature": 9.0},
                ],
            ),
        )
        assert r.status_code == 200, r.text
        cls.digest_b1 = r.json()["analysis_digest"]

        # PREV-B2：校准后 9.5 / 10.0 → 1 条异常 [0,60]
        r = cls.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B2",
                samples=[
                    {"timestamp": iso(0), "temperature": 9.0},
                    {"timestamp": iso(60), "temperature": 9.5},
                ],
            ),
        )
        assert r.status_code == 200, r.text
        cls.digest_b2 = r.json()["analysis_digest"]

        # PREV-B3：校准后 5.5 / 6.5 → 无异常
        r = cls.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B3",
                samples=[
                    {"timestamp": iso(0), "temperature": 5.0},
                    {"timestamp": iso(60), "temperature": 6.0},
                ],
            ),
        )
        assert r.status_code == 200, r.text
        cls.digest_b3 = r.json()["analysis_digest"]

        # 未绑定档案的批次 / 绑定其他档案的批次（用于"档案不匹配"用例）
        r = cls.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B-RAW",
                samples=[{"timestamp": iso(0), "temperature": 9.0}],
                profile_id=None,
            ),
        )
        assert r.status_code == 200, r.text
        r = cls.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B-OTHER",
                samples=[{"timestamp": iso(0), "temperature": 9.0}],
                profile_id="CAL-OTHER",
            ),
        )
        assert r.status_code == 200, r.text

        # 处置：B1 resolved(closed)、B2 accepted(open)、B3 空处置(closed)
        r = cls.client.post(
            "/api/v1/cases/review",
            json={
                "batch_id": "PREV-B1",
                "case_id": "CASE-PREV-1",
                "analysis_digest": cls.digest_b1,
                "decisions": [
                    {"seq": 0, "decision": "resolved",
                     "handled_at": iso(120), "note": "已复冷"},
                ],
            },
        )
        assert r.status_code == 200, r.text
        r = cls.client.post(
            "/api/v1/cases/review",
            json={
                "batch_id": "PREV-B2",
                "case_id": "CASE-PREV-2",
                "analysis_digest": cls.digest_b2,
                "decisions": [
                    {"seq": 0, "decision": "accepted",
                     "handled_at": iso(60)},
                ],
            },
        )
        assert r.status_code == 200, r.text
        r = cls.client.post(
            "/api/v1/cases/review",
            json={
                "batch_id": "PREV-B3",
                "case_id": "CASE-PREV-3",
                "analysis_digest": cls.digest_b3,
                "decisions": [],
            },
        )
        assert r.status_code == 200, r.text

        # 联单：SHP-PREV-1 含 B1+B2（held）；SHP-PREV-2 再次含 B1（released）
        r = cls.client.post(
            "/api/v1/shipments/snapshot",
            json={
                "shipment_id": "SHP-PREV-1",
                "batches": [
                    {"batch_id": "PREV-B1", "analysis_digest": cls.digest_b1,
                     "case_id": "CASE-PREV-1", "handoff_at": iso(1800)},
                    {"batch_id": "PREV-B2", "analysis_digest": cls.digest_b2,
                     "case_id": "CASE-PREV-2", "handoff_at": iso(2400)},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["result"] == "held"
        r = cls.client.post(
            "/api/v1/shipments/snapshot",
            json={
                "shipment_id": "SHP-PREV-2",
                "batches": [
                    {"batch_id": "PREV-B1", "analysis_digest": cls.digest_b1,
                     "case_id": "CASE-PREV-1", "handoff_at": iso(3000)},
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["result"] == "released"

        # 绑定"其他摘要"（过期口径）的案件与联单：绝不出现在受影响清单
        with sqlite3.connect(_TMP_DB) as conn:
            conn.execute(
                """INSERT INTO cases
                   (case_id, batch_id, analysis_digest, content_hash,
                    status, created_at)
                   VALUES ('CASE-PREV-STALE', 'PREV-B1', 'stale-digest',
                           'x', 'closed', 0)"""
            )
            conn.execute(
                """INSERT INTO shipments
                   (shipment_id, content_hash, result, created_at)
                   VALUES ('SHP-PREV-STALE', 'x', 'released', 0)"""
            )
            conn.execute(
                """INSERT INTO shipment_batches
                   (shipment_id, seq, batch_id, case_id, analysis_digest,
                    handoff_at, anomaly_count, case_status)
                   VALUES ('SHP-PREV-STALE', 0, 'PREV-B1', 'CASE-PREV-STALE',
                           'stale-digest', 0, 1, 'closed')"""
            )
            # 设备与档案不匹配的批次（正常流程无法写入，直接插库构造）
            conn.execute(
                """INSERT INTO batches
                   (batch_id, device_id, profile_id, request_hash,
                    request_json, anomaly_count, created_at)
                   VALUES ('PREV-B-DEV', 'reefer-other', ?, 'x', '{}', 0, 0)""",
                (PROFILE,),
            )

    # ---------- 工具 ----------

    def _dump_all(self):
        dump = {}
        with sqlite3.connect(_TMP_DB) as conn:
            for table in TABLES:
                dump[table] = conn.execute(
                    f"SELECT * FROM {table} ORDER BY rowid"
                ).fetchall()
        return dump

    # ---------- 正常预演 ----------

    def test_01_changed_coefficients_marks_all_and_lists_affected(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-B2", "PREV-B3"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 2.0},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["profile_id"], PROFILE)
        self.assertEqual(body["device_id"], DEVICE)
        # 按提交顺序逐批次返回
        self.assertEqual(
            [b["batch_id"] for b in body["batches"]],
            ["PREV-B1", "PREV-B2", "PREV-B3"],
        )

        b1, b2, b3 = body["batches"]
        # B1：+2.0 后 10.0 / 7.0(回落) / 11.0，仍 1 条异常，但摘要必变
        self.assertEqual(b1["anomaly_count_current"], 1)
        self.assertEqual(b1["anomaly_count_proposed"], 1)
        self.assertEqual(b1["analysis_digest_current"], self.digest_b1)
        self.assertNotEqual(
            b1["analysis_digest_proposed"], b1["analysis_digest_current"]
        )
        self.assertTrue(b1["digest_changed"])
        # 绑定旧摘要的案件与联单（排序去重；过期摘要的 CASE/SHP-PREV-STALE 不在内）
        self.assertEqual(b1["affected_case_ids"], ["CASE-PREV-1"])
        self.assertEqual(
            b1["affected_shipment_ids"], ["SHP-PREV-1", "SHP-PREV-2"]
        )

        self.assertEqual(b2["anomaly_count_current"], 1)
        self.assertEqual(b2["anomaly_count_proposed"], 1)
        self.assertTrue(b2["digest_changed"])
        self.assertEqual(b2["affected_case_ids"], ["CASE-PREV-2"])
        self.assertEqual(b2["affected_shipment_ids"], ["SHP-PREV-1"])

        # B3 无异常：两份摘要仍因档案内容不同而变化，案件列入、联单为空
        self.assertEqual(b3["anomaly_count_current"], 0)
        self.assertEqual(b3["anomaly_count_proposed"], 0)
        self.assertTrue(b3["digest_changed"])
        self.assertEqual(b3["affected_case_ids"], ["CASE-PREV-3"])
        self.assertEqual(b3["affected_shipment_ids"], [])

    def test_02_identical_intervals_no_change(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-B3"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        for item in r.json()["batches"]:
            self.assertFalse(item["digest_changed"])
            self.assertEqual(
                item["analysis_digest_proposed"],
                item["analysis_digest_current"],
            )
            self.assertEqual(item["affected_case_ids"], [])
            self.assertEqual(item["affected_shipment_ids"], [])

    def test_03_same_anomalies_but_profile_content_changed_still_marked(self):
        """异常列表未变（区间拆分但系数相同），档案内容改变仍须置变化标记。"""
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                intervals=[
                    # 与原档案系数相同，仅拆成首尾相接的两段 → 校准结果不变
                    {"start": iso(0), "end": iso(900),
                     "scale": 1.0, "offset": 0.5},
                    {"start": iso(900), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        (b1,) = r.json()["batches"]
        # 异常数未变，但档案内容进入摘要 → 摘要变化并列出受影响对象
        self.assertEqual(b1["anomaly_count_current"], 1)
        self.assertEqual(b1["anomaly_count_proposed"], 1)
        self.assertTrue(b1["digest_changed"])
        self.assertNotEqual(
            b1["analysis_digest_proposed"], b1["analysis_digest_current"]
        )
        self.assertEqual(b1["affected_case_ids"], ["CASE-PREV-1"])
        self.assertEqual(
            b1["affected_shipment_ids"], ["SHP-PREV-1", "SHP-PREV-2"]
        )

    def test_04_anomaly_count_changes_with_proposed_intervals(self):
        # scale 0.5：B1 → 4.1/2.6/4.6、B2 → 4.6/4.85，全部回落到限内
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-B2"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 0.5, "offset": 0.1},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        b1, b2 = r.json()["batches"]
        self.assertEqual(b1["anomaly_count_current"], 1)
        self.assertEqual(b1["anomaly_count_proposed"], 0)
        self.assertTrue(b1["digest_changed"])
        self.assertEqual(b2["anomaly_count_current"], 1)
        self.assertEqual(b2["anomaly_count_proposed"], 0)
        self.assertTrue(b2["digest_changed"])

    def test_05_digest_current_matches_live_recompute(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-B2", "PREV-B3"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 2.0},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        for item in r.json()["batches"]:
            live = self.client.get(
                f"/api/v1/batches/{item['batch_id']}"
            ).json()
            self.assertEqual(
                item["analysis_digest_current"], live["analysis_digest"]
            )
            self.assertEqual(
                item["anomaly_count_current"], live["anomaly_count"]
            )

    # ---------- 整次拒绝 ----------

    def test_06_unknown_profile_409(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
                profile_id="CAL-MISSING",
            ),
        )
        self.assertEqual(r.status_code, 409)
        self.assertIn("CAL-MISSING", r.json()["detail"])

    def test_07_unknown_batch_404(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-MISSING"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("PREV-MISSING", r.json()["detail"])

    def test_08_batch_without_profile_409(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B-RAW"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 409)
        self.assertIn("PREV-B-RAW", r.json()["detail"])

    def test_09_batch_bound_to_other_profile_409(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B-OTHER"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertIn("PREV-B-OTHER", detail)
        self.assertIn("CAL-OTHER", detail)

    def test_10_device_mismatch_409(self):
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B-DEV"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 409)
        detail = r.json()["detail"]
        self.assertIn("PREV-B-DEV", detail)
        self.assertIn("reefer-other", detail)

    def test_11_proposed_intervals_missing_sample_422(self):
        # 拟定区间 [0,100)：PREV-B1 在 ts=120 的采样无覆盖 → 整次拒绝
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                intervals=[
                    {"start": iso(0), "end": iso(100),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 422)
        detail = r.json()["detail"]
        self.assertIn("PREV-B1", detail)
        self.assertIn(iso_out(120), detail)

    def test_12_request_validation_422(self):
        valid_intervals = [
            {"start": iso(0), "end": iso(1800), "scale": 1.0, "offset": 0.5}
        ]
        # 批次号重复
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(["PREV-B1", "PREV-B1"], valid_intervals),
        )
        self.assertEqual(r.status_code, 422)
        # 空批次清单
        r = self.client.post(PREVIEW_URL, json=preview_body([], valid_intervals))
        self.assertEqual(r.status_code, 422)
        # 超过 20 个批次
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body([f"B-{i}" for i in range(21)], valid_intervals),
        )
        self.assertEqual(r.status_code, 422)
        # 区间重叠
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                [
                    {"start": iso(0), "end": iso(1000),
                     "scale": 1.0, "offset": 0.5},
                    {"start": iso(900), "end": iso(1800),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 422)
        # end <= start
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                [{"start": iso(600), "end": iso(600),
                  "scale": 1.0, "offset": 0.5}],
            ),
        )
        self.assertEqual(r.status_code, 422)
        # 非正系数
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                [{"start": iso(0), "end": iso(1800),
                  "scale": 0.0, "offset": 0.5}],
            ),
        )
        self.assertEqual(r.status_code, 422)
        # 空区间清单
        r = self.client.post(PREVIEW_URL, json=preview_body(["PREV-B1"], []))
        self.assertEqual(r.status_code, 422)

    # ---------- 只读保证 ----------

    def test_13_preview_is_readonly(self):
        before = self._dump_all()
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1", "PREV-B2", "PREV-B3"],
                intervals=[
                    {"start": iso(0), "end": iso(1800),
                     "scale": 1.0, "offset": 2.0},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self._dump_all(), before)
        # 失败的预演同样不落任何数据
        r = self.client.post(
            PREVIEW_URL,
            json=preview_body(
                ["PREV-B1"],
                intervals=[
                    {"start": iso(0), "end": iso(100),
                     "scale": 1.0, "offset": 0.5},
                ],
            ),
        )
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self._dump_all(), before)

    def test_14_preview_does_not_create_idempotency_records(self):
        # 预演后原接口幂等行为不变：重复分析仍 HIT、案件/联单回放仍可用
        r = self.client.post(
            "/api/v1/batches/analyze",
            json=analysis_body(
                "PREV-B1",
                samples=[
                    {"timestamp": iso(0), "temperature": 8.0},
                    {"timestamp": iso(60), "temperature": 5.0},
                    {"timestamp": iso(120), "temperature": 9.0},
                ],
            ),
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")
        self.assertEqual(r.json()["analysis_digest"], self.digest_b1)
        r = self.client.post(
            "/api/v1/shipments/snapshot",
            json={
                "shipment_id": "SHP-PREV-2",
                "batches": [
                    {"batch_id": "PREV-B1", "analysis_digest": self.digest_b1,
                     "case_id": "CASE-PREV-1", "handoff_at": iso(3000)},
                ],
            },
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("X-Cache"), "HIT")
        self.assertEqual(r.json()["result"], "released")


if __name__ == "__main__":
    unittest.main()
