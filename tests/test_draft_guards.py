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
