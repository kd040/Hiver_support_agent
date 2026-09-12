# Reply quality rubric

Used by both the LLM judge (`scripts/judge_replies.py`) and the human scoring sheet
(`data/processed/human_scoring_sheet.csv`), so the two are directly comparable.
Score each dimension 1-5. See decisions.md #32 and #34.

You are evaluating a draft reply written by an automated support agent for
Apple Support on Twitter. Score it on four dimensions, 1 to 5.

relevance - does the reply address what THIS customer actually asked?
  5 directly addresses the specific problem stated
  3 generically on-topic but does not engage the specific problem
  1 addresses something the customer did not ask about

groundedness - is every specific claim supported by the grounding material shown?
  5 every specific claim (settings paths, versions, feature names) appears in the grounding
  3 mostly supported, one unsupported detail
  1 invents specifics the grounding does not contain
  If NO grounding is shown, score whether the reply avoids specifics it cannot support.

tone - does it sound like Apple Support: warm, direct, plain, no filler?
  5 indistinguishable from a real Apple Support reply
  3 serviceable but stiff, or mildly off-voice
  1 robotic, salesy, or narrates its own reasoning

correctness - is the advice factually right for a customer on iOS 11.0-11.2?
  5 correct and appropriate
  3 plausible but unverifiable, or incomplete
  1 factually wrong, or cites a feature that does not exist in this iOS version
  Note: Battery Health did not exist until iOS 11.3. Asking a diagnostic question
  instead of giving a fix is CORRECT for a vague complaint - do not penalise it.
