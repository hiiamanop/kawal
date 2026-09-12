from contracts.models import Category, PolicyInput, PolicyResult


AUTHORITY_DIRECTORY = {
    ("JUR-FICT-01", Category.ROAD): "UNIT-BINA-MARGA-01",
    ("JUR-FICT-01", Category.DRAINAGE_FLOOD): "UNIT-SDA-01",
    ("JUR-FICT-01", Category.WASTE): "UNIT-LH-01",
    ("JUR-FICT-01", Category.CLEAN_WATER): "UNIT-PDAM-01",
    ("JUR-FICT-01", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-01",
    ("JUR-FICT-01", Category.HEALTH_SERVICE): "UNIT-DINKES-01",

    ("JUR-FICT-02", Category.ROAD): "UNIT-BINA-MARGA-02",
    ("JUR-FICT-02", Category.DRAINAGE_FLOOD): "UNIT-SDA-02",
    ("JUR-FICT-02", Category.WASTE): "UNIT-LH-02",
    ("JUR-FICT-02", Category.CLEAN_WATER): "UNIT-PDAM-02",
    ("JUR-FICT-02", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-02",
    ("JUR-FICT-02", Category.HEALTH_SERVICE): "UNIT-DINKES-02",

    ("JUR-FICT-03", Category.ROAD): "UNIT-BINA-MARGA-03",
    ("JUR-FICT-03", Category.DRAINAGE_FLOOD): "UNIT-SDA-03",
    ("JUR-FICT-03", Category.WASTE): "UNIT-LH-03",
    ("JUR-FICT-03", Category.CLEAN_WATER): "UNIT-PDAM-03",
    ("JUR-FICT-03", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-03",
    ("JUR-FICT-03", Category.HEALTH_SERVICE): "UNIT-DINKES-03",
}


def evaluate_ticket_creation(input_: PolicyInput) -> PolicyResult:
    try:
        try:
            if input_.category not in Category:
                return PolicyResult(
                    decision="DENY",
                    rule_ids=("POL-02",),
                    reason_codes=("AUTHORITY_ROUTE_INVALID",),
                )
        except TypeError:
            return PolicyResult(
                decision="DENY",
                rule_ids=("POL-02",),
                reason_codes=("AUTHORITY_ROUTE_INVALID",),
            )
        expected_unit = AUTHORITY_DIRECTORY.get((input_.jurisdiction_id, input_.category))
        if not expected_unit or expected_unit != input_.authority_unit_id:
            return PolicyResult(
                decision="DENY",
                rule_ids=("POL-02",),
                reason_codes=("AUTHORITY_ROUTE_INVALID",),
            )
        if input_.missing_fields or not input_.has_mandatory_evidence:
            return PolicyResult(
                decision="DENY",
                rule_ids=("POL-03",),
                reason_codes=("MANDATORY_EVIDENCE_MISSING",),
            )
        expected_key = f"{input_.tenant_id}:{input_.case_id}:ticket:create:v1"
        if input_.idempotency_key != expected_key:
            return PolicyResult(
                decision="DENY",
                rule_ids=("POL-06",),
                reason_codes=("IDEMPOTENCY_KEY_INVALID",),
            )
        return PolicyResult(
            decision="ALLOW",
            rule_ids=("POL-02", "POL-03", "POL-06"),
            reason_codes=("TICKET_CREATE_ALLOWED",),
        )
    except Exception:
        return PolicyResult(
            decision="DENY",
            rule_ids=("POL-FAIL-CLOSED",),
            reason_codes=("POLICY_EVALUATION_FAILED",),
        )
