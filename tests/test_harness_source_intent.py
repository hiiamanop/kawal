from __future__ import annotations

from pathlib import Path

from services.harness.source_intent import (
    assess_source_intent,
    has_non_actionable_cue,
)
from services.harness.sources import ReplayXSource


FIXTURES = Path(__file__).parent.parent / "fixtures" / "x_posts"

EXPECTED = {
    "1807221515089506334": "CITIZEN_COMPLAINT",
    "2076110894149652629": "UNKNOWN",
    "931149896203169792": "CITIZEN_COMPLAINT",
    "2026159892885832085": "CITIZEN_COMPLAINT",
    "2125560260849193400": "UNKNOWN",
    "2116307635217838493": "CITIZEN_COMPLAINT",
    "2122148769735360583": "NEWS_REPORT",
    "2121239507798499782": "NEWS_REPORT",
    "2121593107276030456": "NEWS_REPORT",
    "2119723067408879971": "NEWS_REPORT",
    "2115038297194217895": "ANNOUNCEMENT",
    "2067932765530775666": "ANNOUNCEMENT",
    "1998677956323848429": "OPINION",
    "1513890410447441930": "NEWS_REPORT",
    "2099144598069424348": "OPINION",
    "1864479143263367405": "QUESTION",
    "1879519855952511237": "QUESTION",
    "2123836905657921737": "ANNOUNCEMENT",
    "2067364825471361417": "NEWS_REPORT",
    "2066315447985910003": "NEWS_REPORT",
    "2109917602961354968": "ANNOUNCEMENT",
    "2071102573483053232": "UNKNOWN",
    "1925164524103151818": "CITIZEN_COMPLAINT",
    "2060576242433450181": "CITIZEN_COMPLAINT",
    "2107732996271001795": "CITIZEN_COMPLAINT",
    "2120782368684556339": "NEWS_REPORT",
    "2122292838826398059": "NEWS_REPORT",
    "2061727987591245863": "NEWS_REPORT",
    "2054402389671543238": "NEWS_REPORT",
}


def test_source_intent_matches_labeled_fixtures() -> None:
    posts = {post.post_id: post for post in ReplayXSource(FIXTURES).fetch(limit=100)}
    assert set(posts) == set(EXPECTED)
    for post_id, expected in EXPECTED.items():
        post = posts[post_id]
        assessment = assess_source_intent(post.text, post.author_handle)
        assert assessment.intent == expected, (post_id, assessment)


def test_complaint_detail_does_not_trigger_cache_non_actionable_guard() -> None:
    assert (
        has_non_actionable_cue(
            "Jalan rusak tanpa rambu peringatan, tolong segera diperbaiki"
        )
        is False
    )
