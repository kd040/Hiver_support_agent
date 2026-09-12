"""Regression cases for the cleaning/tagging heuristics in scripts/clean_threads.py.

Every case here is a real reply from the AppleSupport corpus, plus the two bugs
that shipped and had to be found by eye.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clean_threads import (  # noqa: E402
    anonymize,
    classify_resolution_type,
    has_bug_keyword,
    in_incident_window,
    tag_incident_window,
)

DM_LINK = "https://t.co/GDrqU22YpT"
HANDLES = {"AppleSupport", "SpotifyCares"}
ALIAS = "115858"


# --- bug #1: a bare U+FE0F matched every VS16 emoji, not the iOS artifact ----
@pytest.mark.parametrize("text", [
    "feeling ☺️ today",
    "thanks ❤️",
    "@BRAND sort it out you dicks 🖕🏻",
    "my phone is slow since the update",
])
def test_vs16_emoji_is_not_the_ios_bug(text):
    assert has_bug_keyword(text) is False


@pytest.mark.parametrize("text", [
    "@BRAND FIX THIS I️ BULLSHIT",                    # I + VS16
    "my phone keeps autocorrecting to I.T.",
    "@BRAND y’all really gotta fix this keyboard glitch",
    "I see the question mark in a box when I type",
    "iOS 11.1.1 did not fix it",
])
def test_real_bug_signals_are_caught(text):
    assert has_bug_keyword(text) is True


# --- bug #2: answer + DM push must not collapse to plain self_contained_answer
def test_real_link_plus_dm_is_answer_with_dm_followup():
    reply = ("@USER Have you updated to 11.1 yet? https://t.co/80YRnjDFDk "
             f"Send us a DM and let us know more details. {DM_LINK}")
    assert classify_resolution_type(reply) == "answer_with_dm_followup"


def test_real_link_without_dm_stays_self_contained():
    reply = ("@USER Here’s what you can do to work around the issue until it’s "
             "fixed in a future software update: https://t.co/xXaXeeSRt9")
    assert classify_resolution_type(reply) == "self_contained_answer"


# --- one clean example per resolution_type ----------------------------------
@pytest.mark.parametrize("expected,reply", [
    ("dm_handoff",
     "@USER We're here to help. Please DM us via the following link and we'll "
     f"assist you further from there. {DM_LINK}"),
    ("self_contained_answer",
     "@USER It sounds like they may be auto connecting if your home network signal "
     "becomes weak. Have you forgotten those Wi-Fi networks in Settings > Wi-Fi? "
     "Give that a try and let us know if you stay connected after."),
    ("answer_with_dm_followup",
     "@USER We’re here for you. Try this: https://t.co/xU1AGHStV6 Should it persist, "
     f"send us a DM to discuss it further. {DM_LINK}"),
    ("clarifying_question",
     "@USER We can assist you. To determine the steps we'll take, can you tell us if "
     "you're using Wi-Fi or iTunes to update your iPhone?"),
    ("other_channel_redirect",
     "@USER Thanks for reaching out! Our Apple TV experts can look into this with you. "
     f"You can reach them here: https://t.co/IBIY3vMgPj {DM_LINK}"),
    ("other", "@USER You're very welcome."),
])
def test_resolution_type_examples(expected, reply):
    assert classify_resolution_type(reply) == expected


def test_bare_dm_deeplink_without_the_word_dm_is_still_a_handoff():
    assert classify_resolution_type(f"@USER We can help you out. {DM_LINK}") == "dm_handoff"


# --- "team up" must not read as a redirect to "our ... team" ----------------
def test_team_up_is_not_a_channel_redirect():
    reply = ("@USER We rely on our Contacts, too, so we'd love to team up and help you "
             f"with this. Let us know in DM and we'll get started. {DM_LINK}")
    assert classify_resolution_type(reply) == "dm_handoff"


# --- anonymization ----------------------------------------------------------
def test_anonymize_maps_handle_and_numeric_alias_to_brand():
    got = anonymize("Dear @AppleSupport and @115858, my @117735 order", ALIAS, HANDLES)
    assert got == "Dear @BRAND and @BRAND, my @USER order"


def test_anonymize_preserves_sentence_structure():
    # Deleting mentions used to leave "Dear , how the hell..."
    assert anonymize("Dear @115858, how are you", ALIAS, HANDLES).startswith("Dear @BRAND,")


# --- incident window --------------------------------------------------------
def test_window_is_inclusive_at_both_ends():
    assert in_incident_window("2017-11-02 00:00:00+00:00") is True
    assert in_incident_window("2017-11-12 23:59:00+00:00") is True
    assert in_incident_window("2017-11-01 23:59:00+00:00") is False
    assert in_incident_window("2017-11-13 00:00:00+00:00") is False


def test_tag_is_an_or_of_date_and_keyword():
    off_topic = "@BRAND where is my order"
    assert tag_incident_window(off_topic, "2017-11-05") == "ios_11_1_bug"   # date only
    assert tag_incident_window("the I️ bug", "2017-09-20") == "ios_11_1_bug"  # keyword only
    assert tag_incident_window(off_topic, "2017-09-20") is None


# --- intent labeling (scripts/label_intents.py) -----------------------------
from label_intents import label_intent, is_non_english  # noqa: E402


@pytest.mark.parametrize("expected,text", [
    ("battery_drain", "@BRAND iOS 11 is killing my battery, went from 85% to 11% overnight"),
    ("billing_account", "@BRAND I've just subbed to Apple Music and the payment has been "
                        "taken but it says I'm not subscribed"),
    ("ios_version_downgrade", "@BRAND How do I get iOS 10 back on my 6s?"),
    ("software_feature_defect", "@BRAND my Messages app keeps crashing since the update"),
    ("general_complaint_nonactionable", "@BRAND this update sucks, sort it out"),
    ("non_english", "@BRAND la actualización al iOS 11.0.3 acabó con mi iPhone 6, "
                    "díganme ustedes qué hacer"),
])
def test_label_intent_examples(expected, text):
    assert label_intent(text) == expected


def test_apostrophe_ve_is_not_turkish():
    # "\\bve\\b" used to match the "ve" in "I've" and route English to non_english.
    assert is_non_english("@BRAND I've been locked out of my phone by a software bug") is False


def test_brand_language_redirect_reply_labels_non_english():
    # Apple's own template is the highest-precision non-English signal available.
    assert label_intent(
        "@BRAND ayuda por favor",
        "@USER We offer support via Twitter in English. Contact us for help in your "
        "preferred language here: https://t.co/IBIY3vMgPj",
    ) == "non_english"


def test_apple_pay_is_a_feature_not_a_purchase():
    assert label_intent("@BRAND Apple Pay won't work on my iPhone X") == "software_feature_defect"


def test_non_english_wins_over_topic_keywords():
    # An always-escalate route must not be overridden by a battery/billing keyword.
    assert label_intent("Ajeita essa bateria aí! Tô carregando meu telefone 3 vezes num dia!") \
        == "non_english"
