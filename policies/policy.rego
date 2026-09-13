package kawal.policy

import future.keywords.if

default allow := false

valid_categories := {
    "ROAD",
    "DRAINAGE_FLOOD",
    "WASTE",
    "CLEAN_WATER",
    "CIVIL_ADMIN",
    "HEALTH_SERVICE",
    "PUBLIC_ORDER",
    "TRANSPORTATION",
    "FIRE_RESCUE",
    "SOCIAL_AFFAIRS",
    "EDUCATION",
    "PARKS_HOUSING",
}

valid_category if {
    valid_categories[input.category]
}

authority_key := sprintf("%s_%s", [input.jurisdiction_id, input.category])
expected_authority := data.kawal.directory.authorities[authority_key]
expected_idempotency_key := sprintf("%s:%s:ticket:create:v1", [input.tenant_id, input.case_id])

valid_authority if {
    valid_category
    expected_authority != null
    expected_authority != ""
    input.authority_unit_id == expected_authority
}

valid_evidence if {
    input.has_mandatory_evidence == true
    count(input.missing_fields) == 0
}

valid_idempotency if {
    input.idempotency_key == expected_idempotency_key
}

allow if {
    valid_authority
    valid_evidence
    valid_idempotency
}

result := {
    "decision": "ALLOW",
    "rule_ids": ["POL-02", "POL-03", "POL-06"],
    "reason_codes": ["TICKET_CREATE_ALLOWED"],
    "bundle_version": "m1.v1",
} if allow

result := {
    "decision": "DENY",
    "rule_ids": ["POL-02"],
    "reason_codes": ["AUTHORITY_ROUTE_INVALID"],
    "bundle_version": "m1.v1",
} if not valid_authority

result := {
    "decision": "DENY",
    "rule_ids": ["POL-03"],
    "reason_codes": ["MANDATORY_EVIDENCE_MISSING"],
    "bundle_version": "m1.v1",
} if {
    valid_authority
    not valid_evidence
}

result := {
    "decision": "DENY",
    "rule_ids": ["POL-06"],
    "reason_codes": ["IDEMPOTENCY_KEY_INVALID"],
    "bundle_version": "m1.v1",
} if {
    valid_authority
    valid_evidence
    not valid_idempotency
}
