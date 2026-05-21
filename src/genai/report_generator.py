"""
src/genai/report_generator.py
──────────────────────────────
GenAI-powered emotional report generator.

Given:
  - Predicted emotion label + confidence scores (from the multimodal model)
  - Optional raw text input from the user
  - Optional image path (for context)

Produces:
  - Natural-language emotional description
  - Simple psychological insight
  - Coping / wellbeing recommendation

Uses the Anthropic Claude API via prompt engineering.
Falls back to a rule-based report if the API key is not set.
"""

import os
import json
from datetime import datetime


# Emotion labels and associated psychological profiles
EMOTION_PROFILES = {
    "happy": {
        "description": "positive affect and elevated mood",
        "body_signals": "raised cheeks, relaxed brow, open posture",
        "psychology":   "associated with dopamine/serotonin release; linked to social connection and reward",
        "suggestions":  ["Savour the moment through mindful appreciation",
                         "Share the positive energy — it is contagious",
                         "Use this high-energy state for creative or demanding tasks"]
    },
    "sad": {
        "description": "low affect and reduced energy",
        "body_signals": "drooped corners of mouth, lowered gaze, slumped posture",
        "psychology":   "often signals loss, disappointment, or unmet needs; promotes reflection",
        "suggestions":  ["Allow yourself to feel — suppressing sadness prolongs it",
                         "Reach out to a trusted person",
                         "Gentle activity (walking, journaling) can help shift mood"]
    },
    "angry": {
        "description": "elevated arousal and approach motivation",
        "body_signals": "furrowed brow, clenched jaw, tense shoulders",
        "psychology":   "signals perceived threat or injustice; adrenaline prepares action",
        "suggestions":  ["Pause before responding — 10 slow breaths lower cortisol",
                         "Identify the unmet need beneath the anger",
                         "Physical exercise is an effective discharge for anger energy"]
    },
    "fear": {
        "description": "threat-detection and avoidance motivation",
        "body_signals": "raised brows, wide eyes, pallor, shallow breathing",
        "psychology":   "amygdala-driven survival response; heightens sensory attention",
        "suggestions":  ["Ground yourself: name 5 things you can see right now",
                         "Regulate breathing (4-7-8 technique) to calm the nervous system",
                         "Distinguish real from imagined threat — most fears are anticipatory"]
    },
    "surprise": {
        "description": "brief orienting response to unexpected stimuli",
        "body_signals": "raised brows, open mouth, momentary freeze",
        "psychology":   "neutral valence; quickly resolves to another emotion (joy or fear)",
        "suggestions":  ["Assess the situation before reacting",
                         "Curiosity is the healthy twin of surprise — lean into it",
                         "Use the moment of openness to absorb new information"]
    },
    "neutral": {
        "description": "baseline affective state with no dominant emotion",
        "body_signals": "relaxed facial muscles, steady gaze, upright posture",
        "psychology":   "indicates emotional equilibrium or affect regulation",
        "suggestions":  ["This is a good state for analytical thinking and decision-making",
                         "Check in: are you suppressing something, or genuinely at peace?",
                         "Use the calm to plan or reflect"]
    },
    "disgust": {
        "description": "rejection and avoidance of perceived contaminants or violations",
        "body_signals": "wrinkled nose, raised upper lip, recoil",
        "psychology":   "evolved to prevent ingestion of toxins; extended to moral violations",
        "suggestions":  ["Name what violated your values — clarity reduces disgust intensity",
                         "Distinguish physical from moral disgust; the latter often needs dialogue",
                         "Self-compassion exercises help when disgust is self-directed"]
    }
}


# ─── Rule-based report (no API) ───────────────────────────────────────────────

def generate_rule_based_report(emotion: str, scores: dict,
                                user_text: str = "") -> dict:
    """
    Generate a structured report without calling any API.
    Used as fallback or for offline use.
    """
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    top2 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
    confidence = scores.get(emotion, 0.0)

    report = {
        "timestamp":  datetime.now().isoformat(),
        "emotion":    emotion,
        "confidence": round(confidence, 3),
        "top_emotions": [{"emotion": e, "score": round(s, 3)} for e, s in top2],
        "description": (
            f"The detected emotion is **{emotion}** (confidence: {confidence:.0%}), "
            f"characterised by {profile['description']}. "
            f"Observable physical signals include: {profile['body_signals']}."
        ),
        "psychological_insight": (
            f"From a psychological perspective, {emotion} {profile['psychology']}. "
            + (f"Your text — \"{user_text[:100]}\" — aligns with this emotional tone."
               if user_text else "")
        ),
        "recommendations": profile["suggestions"],
        "source": "rule-based"
    }
    return report


# ─── LLM-powered report (Anthropic Claude) ────────────────────────────────────

def generate_llm_report(emotion: str, scores: dict, user_text: str = "",
                         model: str = "claude-sonnet-4-20250514",
                         max_tokens: int = 600) -> dict:
    """
    Generate a rich emotional report using the Claude API.

    Requires ANTHROPIC_API_KEY environment variable.
    Falls back to rule-based report on API error.
    """
    try:
        import anthropic
    except ImportError:
        print("[GenAI] anthropic package not installed. pip install anthropic")
        return generate_rule_based_report(emotion, scores, user_text)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("[GenAI] ANTHROPIC_API_KEY not set. Using rule-based report.")
        return generate_rule_based_report(emotion, scores, user_text)

    # Build a clean scores summary for the prompt
    scores_str = ", ".join(
        f"{e}: {s:.0%}"
        for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    )
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])

    system_prompt = """You are an expert clinical psychologist and emotional intelligence coach.
Your role is to produce empathetic, scientifically grounded emotional reports.
Always respond ONLY with a valid JSON object — no markdown, no preamble.
The JSON must have exactly these keys:
  description, psychological_insight, physical_signals, recommendations (list of 3 strings), wellbeing_tip
Keep each field concise (2-4 sentences max for strings, one sentence per recommendation)."""

    user_prompt = f"""Detected emotion: {emotion}
Confidence scores: {scores_str}
Physical signals associated with {emotion}: {profile['body_signals']}
User text (if provided): "{user_text}"

Generate a warm, empathetic, non-clinical emotional report for the user.
Return ONLY the JSON object."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw = message.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_data = json.loads(raw)

        return {
            "timestamp":   datetime.now().isoformat(),
            "emotion":     emotion,
            "confidence":  round(scores.get(emotion, 0.0), 3),
            "top_emotions": [
                {"emotion": e, "score": round(s, 3)}
                for e, s in sorted(scores.items(),
                                    key=lambda x: x[1], reverse=True)[:3]
            ],
            **llm_data,
            "source": f"claude:{model}"
        }

    except json.JSONDecodeError as e:
        print(f"[GenAI] JSON parse error: {e}. Using rule-based fallback.")
        return generate_rule_based_report(emotion, scores, user_text)
    except Exception as e:
        print(f"[GenAI] API error: {e}. Using rule-based fallback.")
        return generate_rule_based_report(emotion, scores, user_text)


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_emotion_report(emotion: str, scores: dict,
                              user_text: str = "",
                              use_llm: bool = True,
                              save_path: str = None) -> dict:
    """
    Public interface for report generation.

    Args:
        emotion   : predicted emotion string
        scores    : dict {emotion_name: probability} for all 7 classes
        user_text : raw text input from the user
        use_llm   : whether to call the Claude API
        save_path : optional path to save the JSON report
    Returns:
        report dict
    """
    if use_llm:
        report = generate_llm_report(emotion, scores, user_text)
    else:
        report = generate_rule_based_report(emotion, scores, user_text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[GenAI] Report saved → {save_path}")

    return report


def print_report(report: dict):
    """Pretty-print a report dict to the console."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  EMOTIONAL ANALYSIS REPORT")
    print(f"  {report['timestamp']}")
    print(sep)
    print(f"  Detected Emotion : {report['emotion'].upper()}  "
          f"(confidence: {report['confidence']:.0%})")
    print(f"\n  Top emotions:")
    for e in report.get("top_emotions", []):
        bar = "█" * int(e['score'] * 20)
        print(f"    {e['emotion']:>10}: {bar:<20} {e['score']:.0%}")
    print(f"\n  Description:\n  {report.get('description', '')}")
    print(f"\n  Psychological Insight:\n  {report.get('psychological_insight', '')}")
    if "physical_signals" in report:
        print(f"\n  Physical Signals:\n  {report['physical_signals']}")
    print(f"\n  Recommendations:")
    for i, rec in enumerate(report.get("recommendations", []), 1):
        print(f"    {i}. {rec}")
    if "wellbeing_tip" in report:
        print(f"\n  Wellbeing Tip:\n  {report['wellbeing_tip']}")
    print(f"\n  Source: {report.get('source', 'unknown')}")
    print(sep + "\n")


# ─── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate an emotion report")
    parser.add_argument("--emotion", default="happy",
                        choices=list(EMOTION_PROFILES.keys()))
    parser.add_argument("--text", default="I feel wonderful today!")
    parser.add_argument("--no-llm", action="store_true",
                        help="Use rule-based report (no API call)")
    parser.add_argument("--save", default=None, help="Path to save JSON report")
    args = parser.parse_args()

    # Simulate model output scores
    import random
    random.seed(42)
    emotions = list(EMOTION_PROFILES.keys())
    scores = {e: random.random() for e in emotions}
    scores[args.emotion] = max(scores.values()) + 0.2   # make selected dominant
    total = sum(scores.values())
    scores = {e: s / total for e, s in scores.items()}

    report = generate_emotion_report(
        emotion=args.emotion,
        scores=scores,
        user_text=args.text,
        use_llm=not args.no_llm,
        save_path=args.save or f"outputs/reports/report_{args.emotion}.json"
    )
    print_report(report)
