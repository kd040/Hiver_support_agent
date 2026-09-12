"""First-pass rule labeling of the taxonomy sample against the six locked intents.

ponytail: keyword/rule heuristics, deliberately crude. This is bootstrap labeling
to size the intents, NOT the golden set -- those get hand-labelled.
"""
import re
import sys
import pandas as pd

SAMPLE = "data/processed/taxonomy_sample.pkl"
OUT = "data/processed/taxonomy_sample_labeled.pkl"

INTENTS = [
    "software_feature_defect",
    "ios_version_downgrade",
    "battery_drain",
    "billing_account",
    "general_complaint_nonactionable",
    "non_english",
]

# Apple's own reply when it cannot serve the language -- the brand labels these for us.
LANG_REDIRECT = re.compile(
    r"support via Twitter in English|preferred language|get help in \w+ here", re.I)
FOREIGN = re.compile(
    r"\b(que|n[ãa]o|para|meu|minha|mais|muito|com|voc[êe]|est[áa]|pra|porque|obrigad[oa]"
    r"|mis|muy|pero|gracias|hola|como|cuando|ahora|nada|hace"
    r"|ajuda|favor|ayuda|quiero|necesito|gostaria|preciso|aiuto|per favore"
    r"|ich|nicht|mein[e]?|bitte|und|ist|kein|habe|auch"
    r"|je|mon|les|une|pas|avec|c'est|tr[èe]s"
    r"|bir|çok|için|daha)\b", re.I)
ACCENTED = re.compile(r"[áàâãäéèêëíìîïóòôõöúùûüñçşğı]", re.I)
ENGLISH_STOP = re.compile(
    r"\b(the|and|is|are|my|i|you|to|for|of|it|this|that|in|on|with|have|has|can|why|what"
    r"|not|but|so|me|your|was|get|just|do|does|all|been|when|how|now|new|please|help|fix)\b", re.I)

DOWNGRADE = re.compile(
    r"\brevert|downgrade|roll ?back|go(ing)? back to (the )?(ios ?10|old|previous|last)"
    r"|back to ios ?10|uninstall the (update|ios)|undo the update"
    r"|(return|restore) to (the )?(previous|old|ios ?10)|want ios ?10|bring back ios ?10"
    r"|ios ?10 back|(get|put|have) .{0,20}ios ?10|previous version of (ios|macos)", re.I)
BATTERY = re.compile(r"\bbatter(y|ies)\b|battery drain|drain(s|ing|ed)?\b", re.I)
# "Apple Pay" is a feature, not a purchase -- masked out before billing matching.
APPLE_PAY = re.compile(r"apple ?pay", re.I)
BILLING = re.compile(
    r"itunes|apple music|subscri|purchas|charged|refund|payment|billing|invoice|receipt"
    r"|gift card|apple id|\baccount\b|\bpassword\b|\border(s|ed|ing)?\b|pre-?order|reservation"
    r"|\bpaid\b|\bpay\b|\bcredit\b|\bbought\b|\bbilled\b|icloud storage"
    r"|\bmy (apple )?id\b|locked out|can'?t (log|sign) ?in|two.?factor|verification code", re.I)
COMPONENT = re.compile(
    r"messages|imessage|notification|keyboard|autocorrect|camera|flashlight|wi-?fi|bluetooth"
    r"|siri|facetime|safari|\bmail\b|photos|airdrop|touch ?id|face ?id|screen|speaker"
    r"|microphone|alarm|calendar|maps|apple ?pay|airpods|earpods|headphone|apple watch"
    r"|macbook|imac|ipad|apple tv|touchpad|trackpad|app store|icloud|hotspot|volume"
    r"|brightness|charger|cable|lightning|\bapps?\b|earpiece|podcast|phone calls?|screen lock"
    r"|ringtone|contacts|wallpaper|\bmusic\b|\bvideo|\bsound\b|\bwatch\b|airplay|hey siri", re.I)
SYMPTOM = re.compile(
    r"crash|freez|not working|isn'?t working|won'?t|can'?t|cannot|broken|stuck|glitch|lag"
    r"|\bslow|disconnect|drop(s|ping|ped)?\b|fail|error|\bbug\b|unresponsive|black screen"
    r"|restart|reboot|delay|missing|disappear|no sound|distort|not letting|doesn'?t work"
    r"|will ?not|not (playing|recognis|recogniz|respond|open|load|charg|connect|work|avail)"
    r"|no (wifi|wi-?fi|sound|service|signal|volume)|too quiet|rotat|declin|locked|unable"
    r"|keeps? (turning|shutting|opening|rotating|freez|crash|dropp)|randomly|crackl"
    r"|do(esn|n)'?t work|didn'?t work|no longer works?|messed up|sticky|barely functions?", re.I)


def is_non_english(customer_text, brand_text=""):
    if LANG_REDIRECT.search(brand_text):
        return True
    hits = FOREIGN.findall(customer_text)
    if len(hits) >= 2 or (hits and not ENGLISH_STOP.search(customer_text)):
        return True
    return bool(ACCENTED.search(customer_text)) and not ENGLISH_STOP.search(customer_text)


def label_intent(customer_text, brand_text=""):
    """Order matters: most specific signal wins, vague complaint is the fallback."""
    if is_non_english(customer_text, brand_text):
        return "non_english"
    if DOWNGRADE.search(customer_text):
        return "ios_version_downgrade"
    if BATTERY.search(customer_text):
        return "battery_drain"
    if BILLING.search(APPLE_PAY.sub(" ", customer_text)):
        return "billing_account"
    if COMPONENT.search(customer_text) and SYMPTOM.search(customer_text):
        return "software_feature_defect"
    return "general_complaint_nonactionable"


def main():
    s = pd.read_pickle(SAMPLE)
    s["intent"] = [label_intent(c, b) for c, b in zip(s["customer_text"], s["brand_text"])]
    s.to_pickle(OUT)

    print(f"labeled {len(s)} taxonomy-sample rows -> {OUT}\n")
    print("=== INTENT DISTRIBUTION ===")
    v = s["intent"].value_counts()
    for k in INTENTS:
        n = int(v.get(k, 0))
        print(f"  {k:<34} {n:>4}  {100 * n / len(s):>5.1f}%  {'#' * round(60 * n / v.max())}")

    print("\n=== intent x resolution_type ===")
    print(pd.crosstab(s["intent"], s["resolution_type"]).to_string())
    print("\n=== intent x incident_window ===")
    print(pd.crosstab(s["intent"], s["incident_window"].fillna("none")).to_string())

    n_ex = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for k in INTENTS:
        sub = s[s["intent"] == k]
        print(f"\n{'#' * 78}\n### {k}  (n={len(sub)})\n{'#' * 78}")
        for r in sub.sample(min(n_ex, len(sub)), random_state=0).itertuples():
            print(f"  [{r.resolution_type}] {r.customer_text[:190]}")


if __name__ == "__main__":
    main()
