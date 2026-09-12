"""Pins the two hard rules layered over the qwen2.5:3b labels (decisions.md #12).

Cases are real rows from the taxonomy sample that the unguarded 3B model got wrong.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from llm_relabel import (DOWNGRADE_RE, OUT_OF_SCOPE, TARGET, is_thin,
                         looks_non_english, real_words, resolve)

THIN = [                                    # link/screenshot only -> nothing to classify
    "@BRAND @USER https://t.co/kNKbOwNDmv",
    "@BRAND What to do ? https://t.co/n55Ls5hyM3",
    "Wow! Thanks @BRAND! https://t.co/tQmKt94V64",
    "@BRAND HELP DKSJSN https://t.co/d8uc1HyAPp",
    "@BRAND @BRAND 11.1 bug... https://t.co/lz0YhvAHMC",
    "@BRAND FIX YOUR COMPASS",
]
NOT_THIN = [
    "@BRAND my iPhone 6s keeps freezing and is very slow after the latest iOS 11.0.3 upgrade",
    "Speakerphone on iOS11 sometimes doesn't work, have to do hard reset to get it back again. @BRAND",
]
REAL_DOWNGRADE = [                          # taxonomy.md examples + true positives
    "@BRAND iOS11 seems to be a disaster. Can I revert to 10? Fix coming? Help!",
    "@BRAND How do I get iOS 10 back on my 6s?",
    "Let me go back to ios 10 plz",
    "Please fix ios11 or let us go back to 10",
    "@BRAND iOS 11 sucks can I go back to 10?",
    "Hey @BRAND how do I find previous versions of macOS on AppStore",
    "@BRAND is there any way to downgrade my iPad to iOS 10?",
]
FAKE_DOWNGRADE = [                          # 3B fired ios_version_downgrade on all of these
    "@BRAND @BRAND please send a new update from ios 11 to iphone 6! The phone is horrible.",
    "@BRAND Ok - iOS 11.0.2 has just now showed up on my iPad Air 2",
    "#ios11.0.3 the WORST #iOS in the history of iOS. I literally want to toss my #iPhone 7+",
    "@BRAND Hello Apple, I am halvung a problem with iOS10.",
    "@BRAND IOS11 disabled my device after restart, no wrong passcode attempts. no one can help?",
]


def test_thin_detection():
    for t in THIN:
        assert is_thin(t), t
    for t in NOT_THIN:
        assert not is_thin(t), t


def test_real_words_strips_urls_and_placeholders():
    assert real_words("@BRAND @USER https://t.co/abc #iPhone") == 0


def test_thin_rows_route_to_residual_whatever_the_model_said():
    for t in THIN:
        assert resolve(t, "non_english") == TARGET, t
        assert resolve(t, "ios_version_downgrade") == TARGET, t


def test_downgrade_precondition_accepts_real_requests():
    for t in REAL_DOWNGRADE:
        assert DOWNGRADE_RE.search(t), t
        assert resolve(t, "ios_version_downgrade") == "ios_version_downgrade", t


def test_downgrade_precondition_rejects_generic_version_complaints():
    for t in FAKE_DOWNGRADE:
        assert not DOWNGRADE_RE.search(t), t
        assert resolve(t, "ios_version_downgrade") == TARGET, t


def test_rules_do_not_touch_other_labels():
    t = "@BRAND my Messages app keeps crashing since the update, please fix"
    assert resolve(t, "software_feature_defect") == "software_feature_defect"
    assert resolve(t, "battery_drain") == "battery_drain"


# --- rule-based non_english (replaces the model's label, which scored 0/14) ---
NON_ENGLISH = [                             # real rows the regex labeler tagged
    "@BRAND @BRAND la actualización al iOS 11.0.3 acabó con mi iPhone 6 díganme ustedes qué hacer.",
    "Tirar print com o IOS 11 tá uma merda. Demora um século pra fazer 😡😡 Meliore Apple @BRAND",
    "Meu, to tendo problema com o wifi, n sei se o problema é o iOS 11 ou o iphone 8 @BRAND help me",
    "@BRAND ME AJUDA POR FAVOR",
]
NON_LATIN_SCRIPT = [
    "@BRAND مرحبا لدي مشكلة في هاتفي بعد التحديث",
    "@BRAND Здравствуйте, у меня проблема с телефоном",
    "@BRAND アップデート後に電話が動作しません",
]
ENGLISH_NOT_FOREIGN = [                     # rows the naive version flagged
    "@USER @BRAND Hi..I purchased for a game its says 29.99$..but they took from 34.97$ ؟؟ what should I do",
    "@BRAND @USER nice little apple ID text message going around. Links to a very good fake website",
    "@BRAND 17 hours to update is a little ridiculous don’t ya think? #AintNobodyGotTimeForThat",
    "@BRAND Please change the Arabic font in iOS11 to previous time. You can ask all Arabic user",
]
# The model dumped these in non_english; all are English and out of taxonomy.
OUT_OF_SCOPE_ROWS = [
    "@BRAND has @BRAND started offering International Warranty on #iPhones? If yes, is it applicable in India too?",
    "@BRAND I just got this scam text message, thought you should know, as it's quite convincing",
    "@BRAND I’m assuming this is a fake email... doesn’t look like what you would send out right?",
    "now we have more characters we can add to a tweet. Will it help? feels like 140 was enough.",
]


def test_non_latin_script_is_non_english():
    for t in NON_LATIN_SCRIPT:
        assert looks_non_english(t), t
        assert resolve(t, "general_complaint_nonactionable") == "non_english", t


def test_latin_script_foreign_language_caught_by_function_words():
    for t in NON_ENGLISH:
        assert looks_non_english(t), t


def test_english_never_flagged_as_non_english():
    for t in ENGLISH_NOT_FOREIGN + OUT_OF_SCOPE_ROWS:
        assert not looks_non_english(t), t


def test_arabic_question_mark_alone_is_not_non_english():
    # \u061F sits inside otherwise-English tweets; only letter ranges count.
    assert not looks_non_english("@BRAND what should I do about this charge ؟؟")


def test_short_message_not_judged_on_function_words():
    assert not looks_non_english("FIX YOUR COMPASS")


def test_model_non_english_becomes_out_of_scope_when_text_is_english():
    for t in OUT_OF_SCOPE_ROWS:
        assert resolve(t, "non_english") == OUT_OF_SCOPE, t


def test_thin_row_beats_out_of_scope_reroute():
    # a bare link the model called non_english is unclassifiable, not out_of_scope
    assert resolve("@BRAND @USER https://t.co/kNKbOwNDmv", "non_english") == TARGET


def test_non_english_rule_beats_thin_rule():
    assert resolve("@BRAND ME AJUDA POR FAVOR", "general_complaint_nonactionable") == "non_english"
