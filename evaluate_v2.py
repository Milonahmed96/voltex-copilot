"""
Voltex Contact Centre Co-Pilot
Evaluation v2 — Independent Test Set

20 fresh questions with different phrasing from the golden set.
Tests generalisation — did the optimisations actually improve the system
or did we just overfit to the original 30 questions?

Scoring dimensions: same as evaluate.py
  - retrieval_correct (0/1)
  - response_quality (1-3)
  - confidence_appropriate (0/1)
  - escalation_correct (0/1)
  Max per question: 6
"""

import json
import time
from pathlib import Path
from datetime import datetime
from copilot import VoltexCoPilot

# ─────────────────────────────────────────────
# INDEPENDENT TEST SET — 20 QUESTIONS
# Different phrasing, more colloquial language,
# cross-category queries, and trap questions
# ─────────────────────────────────────────────

INDEPENDENT_SET = [

    # ── CROSS-CATEGORY (queries that span two documents) ──
    {
        "id": "X01",
        "category": "voltcare",
        "difficulty": "medium",
        "query": "My fridge broke down three weeks after I bought it — should I use VoltCare or just return it?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["30 days", "refund", "statutory", "VoltCare"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Cross-category: within 30 days so statutory rights stronger than VoltCare — agent should present both"
    },
    {
        "id": "X02",
        "category": "voltmobile",
        "difficulty": "medium",
        "query": "I dropped my VoltMobile phone and smashed the screen — is that covered anywhere?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["VoltCare", "accidental damage", "Plus", "Complete", "device insurance"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Cross-category: VoltCare AD claim or VoltMobile device insurance — both options should be mentioned"
    },
    {
        "id": "X03",
        "category": "delivery_orders",
        "difficulty": "hard",
        "query": "My new washing machine arrived but it's making a banging noise — do I return it or get it repaired?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["fault", "30 days", "refund", "repair", "statutory"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Cross-category: fault on new delivery — statutory rights apply, present return and repair options"
    },

    # ── VOLTCARE — different phrasing ──────────
    {
        "id": "VC07",
        "category": "voltcare",
        "difficulty": "easy",
        "query": "I spilled coffee on my laptop keyboard — some keys have stopped working",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["liquid damage", "accidental damage", "Plus", "Complete", "Essential"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Liquid damage = accidental damage — covered under Plus/Complete, not Essential"
    },
    {
        "id": "VC08",
        "category": "voltcare",
        "difficulty": "medium",
        "query": "Can I get VoltCare on a laptop I bought second hand from someone else?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["not purchased from Voltex", "not eligible", "transfer", "original purchase"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Trap: second-hand not from Voltex = not eligible. Transfer only works if originally bought from Voltex."
    },
    {
        "id": "VC09",
        "category": "voltcare",
        "difficulty": "hard",
        "query": "My VoltCare payment failed last week and now my tumble dryer has broken — can I still claim?",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["Grace Period", "14 days", "cover continues", "payment"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Grace Period — cover continues for 14 days after payment failure. Critical policy detail."
    },

    # ── RETURNS AND REPAIRS — colloquial phrasing ──
    {
        "id": "RR07",
        "category": "returns_repairs",
        "difficulty": "easy",
        "query": "I got a TV for Christmas and I don't like it — can I swap it for something different?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["Christmas", "31 January", "goodwill", "exchange", "condition"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Christmas extended returns window — up to 31 January. Exchange processed as return + new purchase."
    },
    {
        "id": "RR08",
        "category": "returns_repairs",
        "difficulty": "medium",
        "query": "I paid for my laptop using buy now pay later — how does the refund work if I return it?",
        "expected_source": "repairs_returns_policy.txt",
        "expected_keywords": ["VoltPay", "Duologi", "finance agreement", "cancelled"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "VoltPay return process — finance agreement cancelled, Duologi refunds payments"
    },
    {
        "id": "RR09",
        "category": "returns_repairs",
        "difficulty": "hard",
        "query": "I had my phone screen fixed at a local repair shop and now it's broken again — Voltex said it voids my warranty",
        "expected_source": "voltcare_policy.txt",
        "expected_keywords": ["unauthorised repair", "voided", "tamper", "physical inspection"],
        "should_escalate": True,
        "expected_confidence": "MEDIUM",
        "notes": "Trap: agent cannot void cover on call — only repair team can confirm after inspection. Must escalate."
    },

    # ── PRODUCTS — conversational phrasing ────
    {
        "id": "PR07",
        "category": "products",
        "difficulty": "easy",
        "query": "Is 8GB RAM enough for a new laptop in 2026?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["8GB", "not recommended", "16GB", "sluggish"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Clear FAQ answer — 8GB not recommended for new purchases in 2026"
    },
    {
        "id": "PR08",
        "category": "products",
        "difficulty": "medium",
        "query": "My mum wants a washing machine that doesn't need defrosting — what should she look for?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["frost-free", "No Frost", "Total No Frost", "automatic"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Colloquial phrasing for frost-free feature — query rewriter must translate correctly"
    },
    {
        "id": "PR09",
        "category": "products",
        "difficulty": "medium",
        "query": "I want to watch football on my new TV — what specs should I look for?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["120Hz", "native", "sport", "bright", "motion"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Sport viewing requirements — native 120Hz is the key spec, not motion processing"
    },
    {
        "id": "PR10",
        "category": "products",
        "difficulty": "hard",
        "query": "Will a Chromebook work for my daughter who needs to use Microsoft Office for school?",
        "expected_source": "product_faqs.txt",
        "expected_keywords": ["Chromebook", "web app", "desktop application", "reduced", "Windows"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Nuanced: Office web app works on Chromebook but not full desktop app — important distinction"
    },

    # ── DELIVERY — real-world scenarios ────────
    {
        "id": "DO07",
        "category": "delivery_orders",
        "difficulty": "easy",
        "query": "I ordered a phone online yesterday and changed my mind — how do I cancel?",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["despatched", "Processing", "cancel", "my-orders"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Simple cancellation before despatch"
    },
    {
        "id": "DO08",
        "category": "delivery_orders",
        "difficulty": "medium",
        "query": "The delivery driver left my laptop on my doorstep without knocking — it got rained on",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["safe place", "carrier", "investigation", "damaged"],
        "should_escalate": True,
        "expected_confidence": "MEDIUM",
        "notes": "Safe place delivery resulting in damage — carrier liability issue, needs escalation"
    },
    {
        "id": "DO09",
        "category": "delivery_orders",
        "difficulty": "hard",
        "query": "I need a washing machine urgently — my old one flooded and I have three kids",
        "expected_source": "delivery_orders_policy.txt",
        "expected_keywords": ["urgent", "priority", "earliest", "next-day", "slot"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "Urgent delivery request — agent should check for earliest available slot and mention VoltCare Complete priority"
    },

    # ── VOLTMOBILE — real-world scenarios ──────
    {
        "id": "VM07",
        "category": "voltmobile",
        "difficulty": "easy",
        "query": "I am going to Dubai for two weeks — will my VoltMobile work there?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["Zone A", "UAE", "Roaming Pass", "£12", "14 days"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Dubai = UAE = Zone A roaming pass £12 for 14 days — not EU roaming"
    },
    {
        "id": "VM08",
        "category": "voltmobile",
        "difficulty": "medium",
        "query": "My VoltMobile bill is way higher than usual this month — I don't understand why",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["usage records", "roaming", "premium rate", "third-party", "check"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "Unexpected bill — must pull usage records before confirming or denying any charge"
    },
    {
        "id": "VM09",
        "category": "voltmobile",
        "difficulty": "hard",
        "query": "I want to move to a different network and keep my number — how long does that take and what happens to my VoltMobile account?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["PAC", "working day", "deactivated", "30 days", "port"],
        "should_escalate": False,
        "expected_confidence": "HIGH",
        "notes": "Porting away — PAC within 2 hours, port completes next working day, number preserved 30 days after closure"
    },
    {
        "id": "VM10",
        "category": "voltmobile",
        "difficulty": "hard",
        "query": "I have been with VoltMobile for 3 years and I'm thinking of leaving — can you do anything to keep me?",
        "expected_source": "voltmobile_policy.txt",
        "expected_keywords": ["PAC", "retention", "cancel", "provide"],
        "should_escalate": False,
        "expected_confidence": "MEDIUM",
        "notes": "Retention scenario — must provide PAC if requested, retention offer permitted but not conditional on withholding PAC"
    },
]


# ─────────────────────────────────────────────
# SCORING (same functions as evaluate.py)
# ─────────────────────────────────────────────

def score_retrieval(result: dict, expected_source: str) -> int:
    top_sources = [c["source"] for c in result["retrieved_chunks"][:3]]
    return 1 if expected_source in top_sources else 0


def score_response_quality(response_text: str, expected_keywords: list[str]) -> int:
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
    levels = ["LOW", "MEDIUM", "HIGH"]
    actual_idx   = levels.index(actual)   if actual   in levels else 1
    expected_idx = levels.index(expected) if expected in levels else 1
    return 1 if abs(actual_idx - expected_idx) <= 1 else 0


def score_escalation(actual: bool, expected: bool) -> int:
    return 1 if actual == expected else 0


# ─────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Voltex Co-Pilot — Independent Test Set v2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions: {len(INDEPENDENT_SET)}")
    print("=" * 65)

    copilot = VoltexCoPilot()

    results       = []
    category_scores = {}

    for i, test in enumerate(INDEPENDENT_SET, 1):
        print(f"\n[{i:02d}/{len(INDEPENDENT_SET)}] {test['id']} ({test['difficulty']}) — {test['query'][:65]}...")

        copilot.reset_conversation()

        start_time = time.time()
        result     = copilot.get_response(test["query"])
        elapsed    = round(time.time() - start_time, 2)

        r = result["response"]

        s_retrieval  = score_retrieval(result, test["expected_source"])
        s_quality    = score_response_quality(r.suggested_response, test["expected_keywords"])
        s_confidence = score_confidence(r.confidence, test["expected_confidence"])
        s_escalation = score_escalation(r.escalate, test["should_escalate"])
        total_score  = s_retrieval + s_quality + s_confidence + s_escalation

        record = {
            "id"                 : test["id"],
            "category"           : test["category"],
            "difficulty"         : test["difficulty"],
            "query"              : test["query"],
            "rewritten_query"    : result["rewritten_query"],
            "detected_category"  : result["category"],
            "expected_source"    : test["expected_source"],
            "top_source"         : result["retrieved_chunks"][0]["source"] if result["retrieved_chunks"] else "none",
            "top_similarity"     : result["retrieved_chunks"][0]["similarity"] if result["retrieved_chunks"] else 0,
            "suggested_response" : r.suggested_response,
            "confidence_actual"  : r.confidence,
            "confidence_expected": test["expected_confidence"],
            "escalate_actual"    : r.escalate,
            "escalate_expected"  : test["should_escalate"],
            "escalation_reason"  : r.escalation_reason,
            "score_retrieval"    : s_retrieval,
            "score_quality"      : s_quality,
            "score_confidence"   : s_confidence,
            "score_escalation"   : s_escalation,
            "total_score"        : total_score,
            "max_score"          : 6,
            "elapsed_seconds"    : elapsed,
            "notes"              : test["notes"],
        }

        results.append(record)

        cat = test["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(total_score)

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

        time.sleep(1)

    # ─────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────

    total_possible = len(INDEPENDENT_SET) * 6
    total_achieved = sum(r["total_score"] for r in results)
    overall_pct    = round(total_achieved / total_possible * 100, 1)

    print("\n" + "=" * 65)
    print("INDEPENDENT TEST SET — SUMMARY")
    print("=" * 65)
    print(f"Overall score: {total_achieved}/{total_possible} ({overall_pct}%)")
    print(f"\n{'Compare to golden set: 81.1% (146/180)'}")

    if overall_pct >= 81.1:
        print("✓ System GENERALISES — independent score matches or exceeds golden set")
    elif overall_pct >= 75.0:
        print("~ System PARTIALLY GENERALISES — minor overfitting to golden set phrasing")
    else:
        print("✗ System OVERFITS — significant drop on unseen phrasings, needs more work")

    print("\nBy category:")
    for cat, scores in category_scores.items():
        cat_total = sum(scores)
        cat_poss  = len(scores) * 6
        cat_pct   = round(cat_total / cat_poss * 100, 1)
        bar       = "█" * int(cat_pct / 5) + "░" * (20 - int(cat_pct / 5))
        print(f"  {cat:<20} {bar} {cat_pct}%  ({cat_total}/{cat_poss})")

    print("\nBy difficulty:")
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if not diff_results:
            continue
        diff_total = sum(r["total_score"] for r in diff_results)
        diff_poss  = len(diff_results) * 6
        diff_pct   = round(diff_total / diff_poss * 100, 1)
        print(f"  {diff:<10} {diff_pct}%  ({diff_total}/{diff_poss})")

    # Cross-category performance
    cross = [r for r in results if r["id"].startswith("X")]
    if cross:
        cross_total = sum(r["total_score"] for r in cross)
        cross_poss  = len(cross) * 6
        cross_pct   = round(cross_total / cross_poss * 100, 1)
        print(f"\nCross-category queries: {cross_pct}%  ({cross_total}/{cross_poss})")

    retrieval_rate = round(sum(r["score_retrieval"] for r in results) / len(results) * 100, 1)
    avg_similarity = round(sum(r["top_similarity"] for r in results) / len(results), 3)
    esc_correct    = sum(r["score_escalation"] for r in results)
    avg_time       = round(sum(r["elapsed_seconds"] for r in results) / len(results), 2)

    print(f"\nRetrieval accuracy:       {retrieval_rate}%")
    print(f"Average top-1 similarity: {avg_similarity}")
    print(f"Escalation accuracy:      {esc_correct}/{len(results)} ({round(esc_correct/len(results)*100,1)}%)")
    print(f"Average response time:    {avg_time}s")

    worst = sorted(results, key=lambda x: x["total_score"])[:5]
    print("\nWORST 5 RESPONSES:")
    for w in worst:
        print(f"  {w['id']} ({w['difficulty']}) — score {w['total_score']}/6")
        print(f"    Query: {w['query'][:70]}...")
        print(f"    Note: {w['notes'][:80]}")

    # Save results
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / "eval_results_v2.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp"        : datetime.now().isoformat(),
                "test_set"         : "independent_v2",
                "total_questions"  : len(INDEPENDENT_SET),
                "overall_score"    : overall_pct,
                "retrieval_rate"   : retrieval_rate,
                "avg_similarity"   : avg_similarity,
                "escalation_rate"  : round(esc_correct / len(results) * 100, 1),
                "avg_response_time": avg_time,
                "golden_set_score" : 81.1,
                "generalises"      : overall_pct >= 75.0,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()