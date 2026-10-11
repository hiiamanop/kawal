from contracts.models import Category, PolicyInput, PolicyResult


AUTHORITY_DIRECTORY = {
    ("JUR-FICT-01", Category.ROAD): "UNIT-BINA-MARGA-01",
    ("JUR-FICT-01", Category.DRAINAGE_FLOOD): "UNIT-SDA-01",
    ("JUR-FICT-01", Category.WASTE): "UNIT-LH-01",
    ("JUR-FICT-01", Category.CLEAN_WATER): "UNIT-PDAM-01",
    ("JUR-FICT-01", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-01",
    ("JUR-FICT-01", Category.HEALTH_SERVICE): "UNIT-DINKES-01",
    ("JUR-FICT-01", Category.PUBLIC_ORDER): "UNIT-SATPOL-PP-01",
    ("JUR-FICT-01", Category.TRANSPORTATION): "UNIT-DISHUB-01",
    ("JUR-FICT-01", Category.FIRE_RESCUE): "UNIT-DISKAR-01",
    ("JUR-FICT-01", Category.SOCIAL_AFFAIRS): "UNIT-DINSOS-01",
    ("JUR-FICT-01", Category.EDUCATION): "UNIT-DISDIK-01",
    ("JUR-FICT-01", Category.PARKS_HOUSING): "UNIT-DPKP-01",

    ("JUR-FICT-02", Category.ROAD): "UNIT-BINA-MARGA-02",
    ("JUR-FICT-02", Category.DRAINAGE_FLOOD): "UNIT-SDA-02",
    ("JUR-FICT-02", Category.WASTE): "UNIT-LH-02",
    ("JUR-FICT-02", Category.CLEAN_WATER): "UNIT-PDAM-02",
    ("JUR-FICT-02", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-02",
    ("JUR-FICT-02", Category.HEALTH_SERVICE): "UNIT-DINKES-02",
    ("JUR-FICT-02", Category.PUBLIC_ORDER): "UNIT-SATPOL-PP-02",
    ("JUR-FICT-02", Category.TRANSPORTATION): "UNIT-DISHUB-02",
    ("JUR-FICT-02", Category.FIRE_RESCUE): "UNIT-DISKAR-02",
    ("JUR-FICT-02", Category.SOCIAL_AFFAIRS): "UNIT-DINSOS-02",
    ("JUR-FICT-02", Category.EDUCATION): "UNIT-DISDIK-02",
    ("JUR-FICT-02", Category.PARKS_HOUSING): "UNIT-DPKP-02",

    ("JUR-FICT-03", Category.ROAD): "UNIT-BINA-MARGA-03",
    ("JUR-FICT-03", Category.DRAINAGE_FLOOD): "UNIT-SDA-03",
    ("JUR-FICT-03", Category.WASTE): "UNIT-LH-03",
    ("JUR-FICT-03", Category.CLEAN_WATER): "UNIT-PDAM-03",
    ("JUR-FICT-03", Category.CIVIL_ADMIN): "UNIT-DUKCAPIL-03",
    ("JUR-FICT-03", Category.HEALTH_SERVICE): "UNIT-DINKES-03",
    ("JUR-FICT-03", Category.PUBLIC_ORDER): "UNIT-SATPOL-PP-03",
    ("JUR-FICT-03", Category.TRANSPORTATION): "UNIT-DISHUB-03",
    ("JUR-FICT-03", Category.FIRE_RESCUE): "UNIT-DISKAR-03",
    ("JUR-FICT-03", Category.SOCIAL_AFFAIRS): "UNIT-DINSOS-03",
    ("JUR-FICT-03", Category.EDUCATION): "UNIT-DISDIK-03",
    ("JUR-FICT-03", Category.PARKS_HOUSING): "UNIT-DPKP-03",
}

CATEGORY_DEPARTMENT_NAMES: dict[Category | str, str] = {
    Category.ROAD: "Dinas Sumber Daya Air dan Bina Marga (DSDABM)",
    Category.DRAINAGE_FLOOD: "Dinas Sumber Daya Air dan Bina Marga (DSDABM)",
    Category.WASTE: "Dinas Lingkungan Hidup (DLH)",
    Category.CLEAN_WATER: "Perumda Tirtawening (PDAM)",
    Category.CIVIL_ADMIN: "Dinas Kependudukan dan Pencatatan Sipil (Disdukcapil)",
    Category.HEALTH_SERVICE: "Dinas Kesehatan (Dinkes)",
    Category.PUBLIC_ORDER: "Satuan Polisi Pamong Praja (Satpol PP)",
    Category.TRANSPORTATION: "Dinas Perhubungan (Dishub)",
    Category.FIRE_RESCUE: "Dinas Kebakaran dan Penanggulangan Bencana (Diskar PB)",
    Category.SOCIAL_AFFAIRS: "Dinas Sosial (Dinsos)",
    Category.EDUCATION: "Dinas Pendidikan (Disdik)",
    Category.PARKS_HOUSING: "Dinas Perumahan dan Kawasan Permukiman (DPKP)",
}

UNIT_DEPARTMENT_NAMES: dict[str, str] = {
    "UNIT-SDA-01": "Dinas Sumber Daya Air dan Bina Marga (DSDABM)",
    "UNIT-BINA-MARGA-01": "Dinas Sumber Daya Air dan Bina Marga (DSDABM)",
    "UNIT-LH-01": "Dinas Lingkungan Hidup (DLH)",
    "UNIT-PDAM-01": "Perumda Tirtawening (PDAM)",
    "UNIT-DUKCAPIL-01": "Dinas Kependudukan dan Pencatatan Sipil (Disdukcapil)",
    "UNIT-DINKES-01": "Dinas Kesehatan (Dinkes)",
    "UNIT-SATPOL-PP-01": "Satuan Polisi Pamong Praja (Satpol PP)",
    "UNIT-DISHUB-01": "Dinas Perhubungan (Dishub)",
    "UNIT-DISKAR-01": "Dinas Kebakaran dan Penanggulangan Bencana (Diskar PB)",
    "UNIT-DINSOS-01": "Dinas Sosial (Dinsos)",
    "UNIT-DISDIK-01": "Dinas Pendidikan (Disdik)",
    "UNIT-DPKP-01": "Dinas Perumahan dan Kawasan Permukiman (DPKP)",
}


def resolve_department_name(category: Category | str | None, unit_id: str | None = None) -> str:
    """Resolve human-readable municipal department name from category and technical unit."""
    if unit_id and unit_id in UNIT_DEPARTMENT_NAMES:
        return UNIT_DEPARTMENT_NAMES[unit_id]
    if category is not None:
        cat_key = (
            category
            if isinstance(category, Category)
            else Category(category)
            if category in Category._value2member_map_
            else category
        )
        if cat_key in CATEGORY_DEPARTMENT_NAMES:
            return CATEGORY_DEPARTMENT_NAMES[cat_key]
        for c, name in CATEGORY_DEPARTMENT_NAMES.items():
            if getattr(c, "value", str(c)) == str(category):
                return name
    return unit_id or "Dinas Terkait"


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
