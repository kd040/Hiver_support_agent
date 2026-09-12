"""Guards on the drafting safety machinery (decisions.md #26, #29).

The battery template's correctness cannot be checked by the specificity flag -- for a
canned template the draft is its own grounding, so the flag is vacuous. These asserts
are the actual guarantee, and they pin the exact defects found in the 14-row run:
Battery Health (iOS 11.3+, outside the corpus window) and Low Power Mode stated with
inverted polarity.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import re

from draft_replies import (BANNED_ERA_TERMS, TEMPLATES, check_overrides, era_violations,
                           has_link, strip_urls, unsupported_specifics)

BATTERY = TEMPLATES["battery_drain"]


def test_battery_template_has_no_post_window_features():
    assert era_violations(BATTERY) == [], era_violations(BATTERY)
    assert "battery health" not in BATTERY.lower()


def test_battery_template_states_low_power_mode_ON_not_off():
    # the exact inversion found in 5 of 6 generated drafts
    assert re.search(r"Low Power Mode on", BATTERY, re.I), BATTERY
    assert not re.search(r"Low Power Mode\s+(is\s+)?(turned\s+)?off", BATTERY, re.I)
    assert not re.search(r"(turn|switch)\s+off\s+Low Power Mode", BATTERY, re.I)


def test_battery_template_covers_the_four_verified_steps():
    for step in ["Settings > Battery", "Low Power Mode", "Background App Refresh",
                 "Software Update"]:
        assert step.lower() in BATTERY.lower(), step


def test_battery_template_is_tweet_length_and_linkless():
    assert len(BATTERY) <= 280, len(BATTERY)
    assert not has_link(BATTERY)


def test_era_guard_catches_the_known_offender():
    assert era_violations("check Settings > Battery > Battery Health") == ["Battery Health"]
    assert "Battery Health" in BANNED_ERA_TERMS


def test_overrides_fire_on_the_three_escalate_anyway_classes():
    assert check_overrides("a gold bit on each side has gone black") == ["physical_or_hardware"]
    assert check_overrides("stop charging me for no reason") == ["account_or_payment_action"]
    assert check_overrides("is this a scam email?") == ["phishing_or_scam"]
    assert check_overrides("my Messages app keeps crashing") == []


def test_grounding_urls_are_stripped_before_prompting():
    assert "t.co" not in strip_urls("see https://t.co/abc123 for help")
    assert has_link("https://t.co/x") and not has_link("no link here")


def test_specificity_check_flags_claims_absent_from_grounding():
    assert unsupported_specifics("go to Settings > General > Keyboard", "this may help")
    assert unsupported_specifics("try Settings > Battery", "open Settings > Battery") == []


# --- narration / length guard (decisions.md #30) ---
def test_narration_guard_catches_the_observed_defect():
    from draft_replies import draft_quality_flags
    bad = ("Let's try to draft a reply based on the grounding material. Since the "
           "grounding material mentions a similar issue, the steps to resolve the issue "
           "are not provided.")
    assert "narration" in draft_quality_flags(bad)
    assert draft_quality_flags("Turning Low Power Mode on helps.") == []


def test_length_guard_reports_the_length():
    from draft_replies import draft_quality_flags
    flags = draft_quality_flags("x" * 400)
    assert flags == ["over_length:400"], flags


def test_every_escalate_route_yields_a_nonempty_reason():
    from escalate import escalation_decision
    rows = [
        {"route": "escalate_override", "override": "physical_or_hardware",
         "intent": "battery_drain"},
        {"route": "no_draft_policy", "intent": "non_english"},
        {"route": "no_draft_policy", "intent": "out_of_scope"},
        {"route": "no_usable_grounding", "intent": "software_feature_defect"},
    ]
    for r in rows:
        action, reason = escalation_decision(r)
        assert action == "escalate", r
        assert reason and reason.strip(), r


def test_auto_handle_routes_are_labelled_by_grounding_kind():
    from escalate import escalation_decision
    a, r = escalation_decision({"route": "policy_fact", "intent": "battery_drain"})
    assert (a, r) == ("auto_handle", "canned_template:battery_drain")
    a, r = escalation_decision({"route": "policy_fact", "intent": "ios_version_downgrade"})
    assert r == "stated_policy_fact:ios_version_downgrade"
    a, r = escalation_decision({"route": "grounded", "intent": "battery_drain",
                                "grounded_on": "123", "grounding_rank": 1,
                                "gated_score": 0.71})
    assert a == "auto_handle" and "rank=1" in r and "gated_score=0.71" in r
