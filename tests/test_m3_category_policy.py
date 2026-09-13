from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import unittest

from contracts.models import Category, PolicyInput
from services.core.policy import AUTHORITY_DIRECTORY, evaluate_ticket_creation


class TestM3CategoryPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.jurisdictions = ("JUR-FICT-01", "JUR-FICT-02", "JUR-FICT-03")
        self.categories = (
            Category.ROAD,
            Category.DRAINAGE_FLOOD,
            Category.WASTE,
            Category.CLEAN_WATER,
            Category.CIVIL_ADMIN,
            Category.HEALTH_SERVICE,
            Category.PUBLIC_ORDER,
            Category.TRANSPORTATION,
            Category.FIRE_RESCUE,
            Category.SOCIAL_AFFAIRS,
            Category.EDUCATION,
            Category.PARKS_HOUSING,
        )

    def test_authority_directory_contains_all_twelve_categories_for_all_jurisdictions(self) -> None:
        self.assertEqual(len(Category), 12)
        for jur_id in self.jurisdictions:
            for cat in self.categories:
                key = (jur_id, cat)
                self.assertIn(key, AUTHORITY_DIRECTORY)
                unit_id = AUTHORITY_DIRECTORY[key]
                self.assertTrue(unit_id.startswith("UNIT-"))

    def test_rego_data_json_matches_python_authority_directory(self) -> None:
        data_path = Path(__file__).resolve().parent.parent / "policies" / "data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        rego_authorities = raw_data["kawal"]["directory"]["authorities"]

        self.assertEqual(len(rego_authorities), len(AUTHORITY_DIRECTORY))
        for (jur_id, cat), expected_unit in AUTHORITY_DIRECTORY.items():
            rego_key = f"{jur_id}_{cat.value}"
            self.assertIn(rego_key, rego_authorities)
            self.assertEqual(rego_authorities[rego_key], expected_unit)

    def test_each_category_expected_unit_allows_valid_policy(self) -> None:
        for jur_id in self.jurisdictions:
            for cat in self.categories:
                unit_id = AUTHORITY_DIRECTORY[(jur_id, cat)]
                policy_input = PolicyInput(
                    tenant_id="research",
                    case_id=f"case-{cat.value.lower()}-001",
                    revision=1,
                    category=cat,
                    jurisdiction_id=jur_id,
                    authority_unit_id=unit_id,
                    missing_fields=(),
                    has_mandatory_evidence=True,
                    idempotency_key=f"research:case-{cat.value.lower()}-001:ticket:create:v1",
                )
                result = evaluate_ticket_creation(policy_input)
                self.assertEqual(result.decision, "ALLOW")
                self.assertEqual(result.rule_ids, ("POL-02", "POL-03", "POL-06"))
                self.assertEqual(result.reason_codes, ("TICKET_CREATE_ALLOWED",))

    def test_each_category_invalid_unit_denies_policy(self) -> None:
        for jur_id in self.jurisdictions:
            for cat in self.categories:
                policy_input = PolicyInput(
                    tenant_id="research",
                    case_id=f"case-{cat.value.lower()}-002",
                    revision=1,
                    category=cat,
                    jurisdiction_id=jur_id,
                    authority_unit_id="UNIT-NON-EXISTENT",
                    missing_fields=(),
                    has_mandatory_evidence=True,
                    idempotency_key=f"research:case-{cat.value.lower()}-002:ticket:create:v1",
                )
                result = evaluate_ticket_creation(policy_input)
                self.assertEqual(result.decision, "DENY")
                self.assertEqual(result.rule_ids, ("POL-02",))
                self.assertEqual(result.reason_codes, ("AUTHORITY_ROUTE_INVALID",))

    def test_cross_category_routing_denies(self) -> None:
        for jur_id in self.jurisdictions:
            road_unit = AUTHORITY_DIRECTORY[(jur_id, Category.ROAD)]
            flood_unit = AUTHORITY_DIRECTORY[(jur_id, Category.DRAINAGE_FLOOD)]
            policy_input = PolicyInput(
                tenant_id="research",
                case_id="case-cross-cat-001",
                revision=1,
                category=Category.DRAINAGE_FLOOD,
                jurisdiction_id=jur_id,
                authority_unit_id=road_unit,
                missing_fields=(),
                has_mandatory_evidence=True,
                idempotency_key="research:case-cross-cat-001:ticket:create:v1",
            )
            result = evaluate_ticket_creation(policy_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-02",))
            self.assertEqual(result.reason_codes, ("AUTHORITY_ROUTE_INVALID",))

            reverse_input = PolicyInput(
                tenant_id="research",
                case_id="case-cross-cat-002",
                revision=1,
                category=Category.ROAD,
                jurisdiction_id=jur_id,
                authority_unit_id=flood_unit,
                missing_fields=(),
                has_mandatory_evidence=True,
                idempotency_key="research:case-cross-cat-002:ticket:create:v1",
            )
            reverse_result = evaluate_ticket_creation(reverse_input)
            self.assertEqual(reverse_result.decision, "DENY")
            self.assertEqual(reverse_result.rule_ids, ("POL-02",))
            self.assertEqual(reverse_result.reason_codes, ("AUTHORITY_ROUTE_INVALID",))

    def test_unknown_jurisdiction_denies(self) -> None:
        for cat in self.categories:
            policy_input = PolicyInput(
                tenant_id="research",
                case_id="case-unknown-jur-001",
                revision=1,
                category=cat,
                jurisdiction_id="JUR-UNKNOWN-99",
                authority_unit_id="UNIT-BINA-MARGA-01",
                missing_fields=(),
                has_mandatory_evidence=True,
                idempotency_key="research:case-unknown-jur-001:ticket:create:v1",
            )
            result = evaluate_ticket_creation(policy_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-02",))
            self.assertEqual(result.reason_codes, ("AUTHORITY_ROUTE_INVALID",))

    def test_unknown_category_does_not_route_accidentally(self) -> None:
        unknown_categories = ("PARK", "PUBLIC_TRANSPORT", "SECURITY", "UNKNOWN", "")
        for unk in unknown_categories:
            mock_input = PolicyInput.model_construct(
                tenant_id="research",
                case_id="case-unk-001",
                revision=1,
                category=unk,
                jurisdiction_id="JUR-FICT-01",
                authority_unit_id="UNIT-BINA-MARGA-01",
                missing_fields=(),
                has_mandatory_evidence=True,
                idempotency_key="research:case-unk-001:ticket:create:v1",
            )
            result = evaluate_ticket_creation(mock_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-02",))
            self.assertEqual(result.reason_codes, ("AUTHORITY_ROUTE_INVALID",))

    def test_missing_mandatory_evidence_denies_across_all_categories(self) -> None:
        for cat in self.categories:
            unit_id = AUTHORITY_DIRECTORY[("JUR-FICT-01", cat)]
            policy_input = PolicyInput(
                tenant_id="research",
                case_id=f"case-{cat.value.lower()}-no-ev",
                revision=1,
                category=cat,
                jurisdiction_id="JUR-FICT-01",
                authority_unit_id=unit_id,
                missing_fields=(),
                has_mandatory_evidence=False,
                idempotency_key=f"research:case-{cat.value.lower()}-no-ev:ticket:create:v1",
            )
            result = evaluate_ticket_creation(policy_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-03",))
            self.assertEqual(result.reason_codes, ("MANDATORY_EVIDENCE_MISSING",))

    def test_missing_fields_denies_across_all_categories(self) -> None:
        for cat in self.categories:
            unit_id = AUTHORITY_DIRECTORY[("JUR-FICT-01", cat)]
            policy_input = PolicyInput(
                tenant_id="research",
                case_id=f"case-{cat.value.lower()}-missing-field",
                revision=1,
                category=cat,
                jurisdiction_id="JUR-FICT-01",
                authority_unit_id=unit_id,
                missing_fields=("incident_location",),
                has_mandatory_evidence=True,
                idempotency_key=f"research:case-{cat.value.lower()}-missing-field:ticket:create:v1",
            )
            result = evaluate_ticket_creation(policy_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-03",))
            self.assertEqual(result.reason_codes, ("MANDATORY_EVIDENCE_MISSING",))

    def test_invalid_idempotency_key_denies_across_all_categories(self) -> None:
        for cat in self.categories:
            unit_id = AUTHORITY_DIRECTORY[("JUR-FICT-01", cat)]
            policy_input = PolicyInput(
                tenant_id="research",
                case_id=f"case-{cat.value.lower()}-bad-idemp",
                revision=1,
                category=cat,
                jurisdiction_id="JUR-FICT-01",
                authority_unit_id=unit_id,
                missing_fields=(),
                has_mandatory_evidence=True,
                idempotency_key="malformed-idempotency-key",
            )
            result = evaluate_ticket_creation(policy_input)
            self.assertEqual(result.decision, "DENY")
            self.assertEqual(result.rule_ids, ("POL-06",))
            self.assertEqual(result.reason_codes, ("IDEMPOTENCY_KEY_INVALID",))

    def test_rego_policy_parity_with_python_policy(self) -> None:
        opa_bin = shutil.which("opa")
        if not opa_bin:
            self.skipTest("OPA binary not found")

        repo_root = Path(__file__).resolve().parent.parent
        rego_path = str(repo_root / "policies" / "policy.rego")
        data_path = str(repo_root / "policies" / "data.json")

        def run_opa(inp: dict) -> dict:
            cmd = [opa_bin, "eval", "-d", rego_path, "-d", data_path, "-I", "data.kawal.policy.result"]
            res = subprocess.run(cmd, input=json.dumps(inp), capture_output=True, text=True, check=True)
            output = json.loads(res.stdout)
            return output["result"][0]["expressions"][0]["value"]

        for jur_id in self.jurisdictions:
            for cat in self.categories:
                unit_id = AUTHORITY_DIRECTORY[(jur_id, cat)]
                valid_input = {
                    "tenant_id": "research",
                    "case_id": f"case-{cat.value.lower()}-rego",
                    "category": cat.value,
                    "jurisdiction_id": jur_id,
                    "authority_unit_id": unit_id,
                    "missing_fields": [],
                    "has_mandatory_evidence": True,
                    "idempotency_key": f"research:case-{cat.value.lower()}-rego:ticket:create:v1",
                }
                rego_res = run_opa(valid_input)
                self.assertEqual(rego_res["decision"], "ALLOW")
                self.assertEqual(rego_res["rule_ids"], ["POL-02", "POL-03", "POL-06"])
                self.assertEqual(rego_res["reason_codes"], ["TICKET_CREATE_ALLOWED"])

                invalid_unit_input = dict(valid_input, authority_unit_id="UNIT-WRONG")
                rego_bad_unit = run_opa(invalid_unit_input)
                self.assertEqual(rego_bad_unit["decision"], "DENY")
                self.assertEqual(rego_bad_unit["rule_ids"], ["POL-02"])
                self.assertEqual(rego_bad_unit["reason_codes"], ["AUTHORITY_ROUTE_INVALID"])

        unknown_cat_input = {
            "tenant_id": "research",
            "case_id": "case-rego-unk",
            "category": "UNKNOWN_CATEGORY",
            "jurisdiction_id": "JUR-FICT-01",
            "authority_unit_id": "UNIT-BINA-MARGA-01",
            "missing_fields": [],
            "has_mandatory_evidence": True,
            "idempotency_key": "research:case-rego-unk:ticket:create:v1",
        }
        rego_unk = run_opa(unknown_cat_input)
        self.assertEqual(rego_unk["decision"], "DENY")
        self.assertEqual(rego_unk["rule_ids"], ["POL-02"])
        self.assertEqual(rego_unk["reason_codes"], ["AUTHORITY_ROUTE_INVALID"])
