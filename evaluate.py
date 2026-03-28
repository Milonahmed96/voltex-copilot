"""
Voltex Contact Centre Co-Pilot
Evaluation Pipeline

Runs 30 golden test questions through the co-pilot,
scores each response, and produces an evaluation report.

Scoring dimensions per question:
  - retrieval_correct (0/1): did the right source appear in top 3 chunks?
  - response_quality (1-3): does the suggested response correctly answer?
      3 = fully correct, specific, grounded in policy
      2 = partially correct or missing a key detail
      1 = incorrect, vague, or potentially harmful
  - confidence_appropriate (0/1): was the confidence level right for this query?
  - escalation_correct (0/1): did escalation trigger correctly (or correctly not trigger)?

Max score per question: 6
Overall score: sum / (30 * 6) as a percentage
"""

import json
import time
from pathlib import Path
from datetime import datetime
from copilot import VoltexCoPilot

# ─────────────────────────────────────────────
# GOLDEN TEST SET — 30 QUESTIONS
# ─────────────────────────────────────────────

GOLDEN_SET = [

    # ── VOLTCARE (6 questions) ──────────────────
    {
        "id": "VC01",
        "category": "voltcare",
        "difficulty": "easy",
        "query": "Does VoltCare cover accidental damage if I drop my phone?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["Plus", "Complete", "accidental damage", "excess"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Clear policy answer — Plus and Complete cover AD, Essential does not"
    },
    {
        "id": "VC02",
        "category": "voltcare",
        "difficulty": "easy",
        "query": "How do I make a VoltCare claim?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["0333 400 7000", "voltex.co.uk", "plan reference", "VC-"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Straightforward claim process question"
    },
    {
        "id": "VC03",
        "category": "voltcare",
        "difficulty": "medium",
        "query": "I bought my VoltCare plan in 2023 — do I still have to pay the excess on accidental damage?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["Version 4.5", "October 2024", "no excess", "before"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "Version history question — pre-Oct 2024 plans have no excess. Agent must check CRM."
    },
    {
        "id": "VC04",
        "category": "voltcare",
        "difficulty": "medium",
        "query": "My OLED TV has burn-in — is that covered under VoltCare Complete?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["burn-in", "not covered", "characteristic", "OLED"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "OLED burn-in is explicitly excluded — common customer misconception"
    },
    {
        "id": "VC05",
        "category": "voltcare",
        "difficulty": "hard",
        "query": "I have VoltCare on my washing machine and it broke down — how long will the repair take and what if they can't fix it?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["15 business days", "engineer", "replacement", "gift card"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Multi-part question covering white goods SLA and replacement trigger"
    },
    {
        "id": "VC06",
        "category": "voltcare",
        "difficulty": "hard",
        "query": "My neighbour repaired my laptop before I knew about VoltCare — will the claim still be valid?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["unauthorised repair", "voided", "tamper", "third party"],
        "should_escalate": True,
        "expected_confidence": "MEDIUM",
        "notes": "Unauthorised repair voids cover — but agent cannot confirm without physical inspection"
    },

    # ── RETURNS AND REPAIRS (6 questions) ──────
    {
        "id": "RR01",
        "category": "returns_repairs",
        "difficulty": "easy",
        "query": "I bought a laptop two weeks ago and it has stopped working — what are my options?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["30 days", "full refund", "statutory", "Consumer Rights Act"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Within 30-day window — statutory right to full refund applies"
    },
    {
        "id": "RR02",
        "category": "returns_repairs",
        "difficulty": "easy",
        "query": "I changed my mind about a TV I bought online three days ago — can I return it?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["14 days", "Consumer Contracts", "distance selling", "cancel"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Online purchase within 14-day cancellation right under Consumer Contracts Regulations"
    },
    {
        "id": "RR03",
        "category": "returns_repairs",
        "difficulty": "medium",
        "query": "My TV arrived damaged — I only noticed after the delivery team left. What happens now?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["48 hours", "photograph", "collection", "replacement"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Damage discovered post-delivery — 48-hour window applies"
    },
    {
        "id": "RR04",
        "category": "returns_repairs",
        "difficulty": "medium",
        "query": "I dropped my laptop and cracked the screen — can I return it for a refund?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["accidental damage", "not a return", "VoltCare", "paid repair"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "AD is not a return — redirect to VoltCare AD claim or paid repair"
    },
    {
        "id": "RR05",
        "category": "returns_repairs",
        "difficulty": "hard",
        "query": "I bought a TV 8 months ago and the screen has developed coloured lines — is Voltex responsible?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["6 months", "burden of proof", "statutory", "manufacturing defect"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "After 6 months burden shifts to customer — but screen fault at 8 months is likely a defect"
    },
    {
        "id": "RR06",
        "category": "returns_repairs",
        "difficulty": "hard",
        "query": "The product listing said this laptop has a dedicated GPU but it only has integrated graphics — I want a full refund",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["not as described", "statutory right", "any time", "Trading Standards"],
        "should_escalate": True,
        "expected_confidence": "HIGH",
        "notes": "Misdescription — statutory right at any time, must escalate to Trading Standards team"
    },

    # ── PRODUCTS (6 questions) ──────────────────
    {
        "id": "PR01",
        "category": "products",
        "difficulty": "easy",
        "query": "What is the difference between OLED and QLED televisions?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["OLED", "QLED", "backlight", "quantum dot", "black levels"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Direct FAQ match — highest similarity expected"
    },
    {
        "id": "PR02",
        "category": "products",
        "difficulty": "easy",
        "query": "How much RAM do I need in a laptop for everyday use?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["16GB", "recommended", "8GB", "minimum"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Direct FAQ match"
    },
    {
        "id": "PR03",
        "category": "products",
        "difficulty": "medium",
        "query": "I want to use my new TV for gaming on PlayStation 5 — what should I look for?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["HDMI 2.1", "120Hz", "VRR", "input lag"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Specific gaming TV requirements — HDMI 2.1 is the key spec"
    },
    {
        "id": "PR04",
        "category": "products",
        "difficulty": "medium",
        "query": "Is a Chromebook suitable for a university student studying engineering?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["Chromebook", "Windows software", "CAD", "not suitable"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Chromebook not suitable for engineering software — important to get right"
    },
    {
        "id": "PR05",
        "category": "products",
        "difficulty": "medium",
        "query": "What washing machine drum size do I need for a family of four?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["10kg", "family", "8kg", "drum size"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Straightforward buying guide question"
    },
    {
        "id": "PR06",
        "category": "products",
        "difficulty": "hard",
        "query": "My laptop battery drains really fast — is that covered under warranty?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["degradation", "wear", "12 months", "charge cycles"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "Battery degradation vs premature failure — nuanced answer needed"
    },

    # ── DELIVERY AND ORDERS (6 questions) ──────
    {
        "id": "DO01",
        "category": "delivery_orders",
        "difficulty": "easy",
        "query": "My washing machine was supposed to be delivered today but it hasn't arrived",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["large item", "track", "my-orders", "rebook"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Missed large item delivery — standard process"
    },
    {
        "id": "DO02",
        "category": "delivery_orders",
        "difficulty": "easy",
        "query": "Can I cancel my order before it is delivered?",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["despatched", "cancel", "Processing", "refund"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Cancellation before despatch is straightforward"
    },
    {
        "id": "DO03",
        "category": "delivery_orders",
        "difficulty": "medium",
        "query": "I need to change my delivery address — the order was placed an hour ago",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["phone", "0333 400 7000", "before carrier", "two systems"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Address change requires phone call and updating two systems"
    },
    {
        "id": "DO04",
        "category": "delivery_orders",
        "difficulty": "medium",
        "query": "I want Voltex to install my new washing machine when they deliver it",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["VoltInstall", "£49", "48 hours", "before delivery"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "VoltInstall must be added 48 hours before delivery"
    },
    {
        "id": "DO05",
        "category": "delivery_orders",
        "difficulty": "hard",
        "query": "I live in the Scottish Highlands — can I get my fridge-freezer delivered and will there be extra charges?",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["Highlands", "surcharge", "£9.99", "extended timescale"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Postcode restriction with surcharge — specific detail required"
    },
    {
        "id": "DO06",
        "category": "delivery_orders",
        "difficulty": "hard",
        "query": "The delivery team said they couldn't get my fridge through the door — what happens now?",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["access", "failed delivery", "£25", "collection", "slimline"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Access failure — failed delivery charge applies, but refund available for access issues"
    },

    # ── VOLTMOBILE (6 questions) ────────────────
    {
        "id": "VM01",
        "category": "voltmobile",
        "difficulty": "easy",
        "query": "I am going to France next month — will I be charged extra to use my phone?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["EU roaming", "60GB", "Roaming Pass", "£10"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "EU roaming — depends on plan tier"
    },
    {
        "id": "VM02",
        "category": "voltmobile",
        "difficulty": "easy",
        "query": "How do I bring my existing phone number to VoltMobile?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["PAC", "65075", "porting", "working day"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Number porting process — clear PAC code instructions"
    },
    {
        "id": "VM03",
        "category": "voltmobile",
        "difficulty": "medium",
        "query": "I have been charged for roaming but I never left the UK",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["border", "Northern Ireland", "ferry", "goodwill"],
        "should_escalate": True,
        "expected_confidence": "MEDIUM",
        "notes": "Border proximity roaming — escalate to TL for goodwill credit assessment"
    },
    {
        "id": "VM04",
        "category": "voltmobile",
        "difficulty": "medium",
        "query": "My phone has been stolen — what should I do right now?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["bar", "SIM", "police", "crime reference", "90 days"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Lost/stolen — bar SIM immediately, advise police report for insurance"
    },
    {
        "id": "VM05",
        "category": "voltmobile",
        "difficulty": "hard",
        "query": "I want to cancel my 24-month phone plan — I have 14 months left and I also owe money on the handset",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["early termination", "0.97", "handset settlement", "both charges"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Must quote BOTH service ETC and handset settlement — critical agent error if only one quoted"
    },
    {
        "id": "VM06",
        "category": "voltmobile",
        "difficulty": "hard",
        "query": "There is a charge on my bill for a third-party service I did not sign up for",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["third-party", "bar", "dispute", "14 to 28 days", "goodwill"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Third-party charge — apply bar immediately, raise dispute, possible goodwill credit"
    },
]


# ─────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────

def score_retrieval(result: dict, expected_source: str) -> int:
    """
    Checks whether the expected source document appears in the top 3 chunks.
    Returns 1 if found, 0 if not.
    """
    top_sources = [c["source"] for c in result["retrieved_chunks"][:3]]
    return 1 if expected_source in top_sources else 0


def score_response_quality(response_text: str, expected_keywords: list[str]) -> int:
    """
    Scores response quality 1-3 based on keyword coverage.
    3 = 75%+ keywords present
    2 = 40-74% keywords present
    1 = under 40% keywords present

    This is a proxy metric. Real evaluation would use human scoring.
    Keywords represent the essential policy concepts the answer must contain.
    """
    response_lower = response_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    coverage = hits / len(expected_keywords) if expected_keywords else 0

    if coverage >= 0.75:
        return 3
    elif coverage >= 0.40:
        return 2
    else:
        return 1


def score_confidence(actual: str, expected: str) -> int:
    """
    Returns 1 if confidence level is appropriate, 0 if clearly wrong.
    Allows one step of slack (MEDIUM accepted where HIGH expected and vice versa).
    """
    levels = ["LOW", "MEDIUM", "HIGH"]
    actual_idx   = levels.index(actual)   if actual   in levels else 1
    expected_idx = levels.index(expected) if expected in levels else 1
    return 1 if abs(actual_idx - expected_idx) <= 1 else 0


def score_escalation(actual: bool, expected: bool) -> int:
    """Returns 1 if escalation matches expected, 0 if not."""
    return 1 if actual == expected else 0


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Voltex Co-Pilot — Evaluation Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions: {len(GOLDEN_SET)}")
    print("=" * 65)

    copilot = VoltexCoPilot()

    results = []
    category_scores = {}

    for i, test in enumerate(GOLDEN_SET, 1):
        print(f"\n[{i:02d}/{len(GOLDEN_SET)}] {test['id']} ({test['difficulty']}) — {test['query'][:60]}...")

        # Reset conversation for each question
        copilot.reset_conversation()

        # Run the query
        start_time = time.time()
        result = copilot.get_response(test["query"])
        elapsed = round(time.time() - start_time, 2)

        r = result["response"]

        # Score each dimension
        s_retrieval   = score_retrieval(result, test["expected_source"])
        s_quality     = score_response_quality(r.suggested_response, test["expected_keywords"])
        s_confidence  = score_confidence(r.confidence, test["expected_confidence"])
        s_escalation  = score_escalation(r.escalate, test["should_escalate"])
        total_score   = s_retrieval + s_quality + s_confidence + s_escalation

        # Build result record
        record = {
            "id"                    : test["id"],
            "category"              : test["category"],
            "difficulty"            : test["difficulty"],
            "query"                 : test["query"],
            "rewritten_query"       : result["rewritten_query"],
            "detected_category"     : result["category"],
            "expected_source"       : test["expected_source"],
            "top_source"            : result["retrieved_chunks"][0]["source"] if result["retrieved_chunks"] else "none",
            "top_similarity"        : result["retrieved_chunks"][0]["similarity"] if result["retrieved_chunks"] else 0,
            "suggested_response"    : r.suggested_response,
            "confidence_actual"     : r.confidence,
            "confidence_expected"   : test["expected_confidence"],
            "escalate_actual"       : r.escalate,
            "escalate_expected"     : test["should_escalate"],
            "escalation_reason"     : r.escalation_reason,
            "score_retrieval"       : s_retrieval,
            "score_quality"         : s_quality,
            "score_confidence"      : s_confidence,
            "score_escalation"      : s_escalation,
            "total_score"           : total_score,
            "max_score"             : 6,
            "elapsed_seconds"       : elapsed,
            "notes"                 : test["notes"],
        }

        results.append(record)

        # Track by category
        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(total_score)

        # Print per-question summary
        quality_label = {3: "✓✓", 2: "✓~", 1: "✗"}.get(s_quality, "?")
        print(
            f"  Score: {total_score}/6 | "
            f"Retrieval: {'✓' if s_retrieval else '✗'} | "
            f"Quality: {quality_label} | "
            f"Conf: {'✓' if s_confidence else '✗'} | "
            f"Escalation: {'✓' if s_escalation else '✗'} | "
            f"sim={record['top_similarity']} | {elapsed}s"
        )

        if s_quality < 3:
            print(f"  Response: {r.suggested_response[:120]}...")

        # Small delay to avoid rate limiting
        time.sleep(1)

    # ─────────────────────────────────────────────
    # SUMMARY STATISTICS
    # ─────────────────────────────────────────────

    total_possible = len(GOLDEN_SET) * 6
    total_achieved = sum(r["total_score"] for r in results)
    overall_pct    = round(total_achieved / total_possible * 100, 1)

    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Overall score: {total_achieved}/{total_possible} ({overall_pct}%)")

    print("\nBy category:")
    for cat, scores in category_scores.items():
        cat_total    = sum(scores)
        cat_possible = len(scores) * 6
        cat_pct      = round(cat_total / cat_possible * 100, 1)
        bar          = "█" * int(cat_pct / 5) + "░" * (20 - int(cat_pct / 5))
        print(f"  {cat:<20} {bar} {cat_pct}%  ({cat_total}/{cat_possible})")

    print("\nBy difficulty:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        diff_total   = sum(r["total_score"] for r in diff_results)
        diff_poss    = len(diff_results) * 6
        diff_pct     = round(diff_total / diff_poss * 100, 1)
        print(f"  {diff:<10} {diff_pct}%  ({diff_total}/{diff_poss})")

    # Retrieval analysis
    retrieval_scores = [r["score_retrieval"] for r in results]
    retrieval_rate   = round(sum(retrieval_scores) / len(retrieval_scores) * 100, 1)
    avg_similarity   = round(sum(r["top_similarity"] for r in results) / len(results), 3)
    print(f"\nRetrieval accuracy: {retrieval_rate}%")
    print(f"Average top-1 similarity: {avg_similarity}")

    # Worst 5 responses
    worst = sorted(results, key=lambda x: x["total_score"])[:5]
    print("\nWORST 5 RESPONSES (lowest scores):")
    for w in worst:
        print(f"  {w['id']} ({w['difficulty']}) — score {w['total_score']}/6")
        print(f"    Query: {w['query'][:70]}...")
        print(f"    Top source: {w['top_source']} (sim={w['top_similarity']})")
        print(f"    Response: {w['suggested_response'][:100]}...")

    # Escalation accuracy
    esc_correct = sum(r["score_escalation"] for r in results)
    print(f"\nEscalation accuracy: {esc_correct}/{len(results)} ({round(esc_correct/len(results)*100,1)}%)")

    # Average response time
    avg_time = round(sum(r["elapsed_seconds"] for r in results) / len(results), 2)
    print(f"Average response time: {avg_time}s")

    # ─────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    # Full results JSON
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp"       : datetime.now().isoformat(),
                "total_questions" : len(GOLDEN_SET),
                "overall_score"   : overall_pct,
                "retrieval_rate"  : retrieval_rate,
                "avg_similarity"  : avg_similarity,
                "avg_response_time": avg_time,
            },
            "results": results,
        }, f, indent=2)

    # Markdown summary report
    report_path = output_dir / "eval_report.md"
    with open(report_path, "w") as f:
        f.write(f"# Voltex Co-Pilot — Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Overall Score\n\n")
        f.write(f"**{overall_pct}%** ({total_achieved}/{total_possible})\n\n")
        f.write(f"## Scores by Category\n\n")
        f.write("| Category | Score | % |\n|---|---|---|\n")
        for cat, scores in category_scores.items():
            ct = sum(scores)
            cp = len(scores) * 6
            f.write(f"| {cat} | {ct}/{cp} | {round(ct/cp*100,1)}% |\n")
        f.write(f"\n## Retrieval Metrics\n\n")
        f.write(f"- Retrieval accuracy (correct source in top 3): **{retrieval_rate}%**\n")
        f.write(f"- Average top-1 similarity score: **{avg_similarity}**\n")
        f.write(f"- Average response time: **{avg_time}s**\n\n")
        f.write(f"## Worst 5 Responses\n\n")
        for w in worst:
            f.write(f"### {w['id']} — {w['query'][:60]}...\n")
            f.write(f"- **Score:** {w['total_score']}/6\n")
            f.write(f"- **Top source:** {w['top_source']} (sim={w['top_similarity']})\n")
            f.write(f"- **Response:** {w['suggested_response'][:200]}\n\n")
        f.write(f"## Escalation Accuracy\n\n")
        f.write(f"**{esc_correct}/{len(results)}** ({round(esc_correct/len(results)*100,1)}%)\n\n")

    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to:  {report_path}")
    print("\nDay 5 Part 1 complete.")


if __name__ == "__main__":
    main()