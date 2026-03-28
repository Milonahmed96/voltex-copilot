# Voltex Co-Pilot — Business Reflection

**Project:** Contact Centre Co-Pilot  
**Author:** Milon Ahmed  
**Date:** March 2026  
**Stack:** Python, ChromaDB, sentence-transformers, Claude Sonnet, Streamlit  
**Evaluation score:** Evaluation score: 82.0% combined across 50 questions (30 golden set + 20 independent). System generalises — independent set scored 83.3% vs 81.1% on tuned golden set.
**Retrival Score:** from 90% to 93.3%. 

---

## What Problem Does This Solve?

Voltex's contact centre handles millions of customer interactions per year across
five distinct domains: VoltCare warranty claims, returns and repairs, product advice,
delivery and orders, and VoltMobile plan support. Each domain has its own policy
document running to thousands of words, with version histories, edge cases, and
interaction effects between categories.

An agent handling 80–100 calls per day cannot hold all of this in working memory.
The result is inconsistency — customers with identical queries receive different
answers depending on which agent they reach. More seriously, agents who don't know
the exact policy make errors that generate complaints, compensation costs, and
regulatory risk. Common examples identified during this project include:

- Incorrectly applying the October 2024 excess charge to pre-October 2024 VoltCare plans
- Denying claims during the 14-day Grace Period when cover legally continues
- Quoting only the service early termination charge on phone plan cancellations
  without mentioning the handset settlement balance
- Failing to present both replacement options (like-for-like or gift card) on
  beyond-economical-repair decisions

The co-pilot addresses this by retrieving the exact relevant policy at the moment
of the call and generating a suggested response grounded in that policy — not in
the agent's memory.

---

## What Did the Evaluation Reveal?

The 30-question golden test set produced an overall score of 78.3% with 90%
retrieval accuracy. The results confirm three things:

**Where the system works well:** straightforward policy lookups (easy questions
scored 86.7%), product knowledge questions (86.1% category score), and VoltCare
queries (86.1%). The FAQ chunking strategy — keeping each Q&A pair as a single
atomic retrieval unit — produced significantly better results than the sliding
window chunker used for policy documents. The OLED vs QLED question returned a
0.823 similarity score, the highest in the evaluation.

**Where the system struggles:** hard questions requiring the agent to synthesise
multiple policy sections scored 68.5%. The worst failures were multi-part questions
(washing machine breakdown timescales plus replacement options) and queries where
the customer's language doesn't match the policy's terminology (battery drains fast
vs battery degradation after charge cycles).

**Where the design decisions proved correct:** query rewriting before retrieval
fixed the most significant weakness identified during Day 2 smoke testing. The
Spain roaming query — which returned 0.407 similarity without rewriting — returned
0.709 after the rewrite step converted "going to Spain" into "EU roaming VoltMobile
phone usage abroad Spain". This single architectural decision accounts for a
meaningful share of the overall score.

---

## What Would This Cost in Production?

At Voltex's contact centre scale, the unit economics matter significantly.

**API cost estimate:**
Each co-pilot query makes two Claude Sonnet API calls — one for query
rewriting and classification (~150 tokens), one for reasoning (~1,000 tokens).
At approximately $0.003 per 1K output tokens for Claude Sonnet, each co-pilot
interaction costs roughly $0.003–$0.005.

At 2,000 agents handling 80 calls per day, with 50% of calls requiring co-pilot
assistance, that is approximately 80,000 queries per day — a daily API cost of
around £250–£320, or roughly £80,000–£100,000 per year.

**Cost vs value:**
Industry benchmarks suggest AI-assisted agents resolve calls 20–35% faster and
with 15–25% fewer escalations. At £5 average cost per contact centre interaction
and 80,000 assisted interactions per day, a 20% handle time reduction represents
potential savings of approximately £2.9 million per year — well above the API cost.

The more important cost metric is complaint and compensation reduction. A single
VoltCare mis-sale or Consumer Rights Act error that reaches the Financial Ombudsman
Service can cost £500–£750 in ombudsman fees alone, plus compensation and internal
handling cost. The co-pilot's policy grounding directly reduces this risk.

---

## What Are the Real Risks?

A system like this deployed in a live contact centre carries three categories
of meaningful risk:

**Policy accuracy risk:** the co-pilot is only as accurate as the knowledge base
it retrieves from. If a policy document is updated but the knowledge base is not
re-indexed, the system will confidently give outdated advice. In this project,
the VoltCare version history (V4.5 vs V4.7) was one of the hardest things to
retrieve correctly. Production deployment requires a process for re-indexing
whenever any policy document changes — not just a technical update, but an
operational workflow owned by a product manager.

**Confidence calibration risk:** the evaluation showed the system sometimes returns
HIGH confidence on questions that require agent judgement (VC03 — the pre-2024
excess question scored MEDIUM confidence correctly, but this is precisely the
kind of question that could mislead an agent who trusts a HIGH confidence response
without verifying the plan version in CRM). Production deployment requires clear
agent training: the co-pilot is a first draft, not a final answer.

**Escalation failure risk:** the evaluation found two escalation misses in 30
questions (93.3% accuracy). The two failures — DO06 (access failure delivery) and
VM03 (border-proximity roaming charge) — are both cases where the system gave a
reasonable partial answer but failed to recognise the need for Team Leader
authorisation. In a production contact centre, a missed escalation on a roaming
dispute or a delivery failure involving a vulnerable customer has real regulatory
consequences. The escalation logic needs to be more conservative, with explicit
triggers for any query involving financial disputes, potential mis-sales, or
vulnerable customer indicators.

---

## What Guardrails Would a Production Version Need?

This prototype has none of the guardrails a production deployment requires:

**Knowledge base versioning:** every document in the knowledge base should carry
a version number and effective date. The retrieval system should surface this
metadata so the agent can see not just what the policy says but which version was
retrieved and when it was last updated.

**Human override logging:** every response the agent accepts, modifies, or
ignores should be logged. This creates the feedback loop needed to improve the
system — agents who consistently override a particular type of response are
signalling a retrieval or reasoning gap.

**Hallucination guardrail:** the system prompt instructs Claude never to invent
policy details, but this is not enforced programmatically. A production version
should compare the key facts in the generated response against the retrieved
chunks and flag any claim that cannot be grounded in a specific chunk.

**PII handling:** the current system logs the full customer query including any
personally identifiable information mentioned. A production system must ensure
queries are not stored in logs beyond what is necessary and are handled in
compliance with UK GDPR.

**Latency SLA:** average response time in evaluation was 6.67 seconds. For a live
call where the customer is waiting, 6–7 seconds is acceptable but not comfortable.
A production version should target under 3 seconds, achievable through caching
frequent query embeddings and using a smaller classification model.

---

## What Would I Build Next?

Three improvements would have the highest impact on the evaluation score:

**1. Re-ranking with a cross-encoder:** the current system uses bi-encoder
embeddings for retrieval (fast but approximate). Adding a cross-encoder re-ranker
as a second pass — scoring the top 10 retrieved chunks against the query for
precise relevance — would improve hard question performance significantly. The
five worst evaluation responses all involved correct retrieval of the right
document but wrong chunk selection within that document.

**2. Structured escalation rules:** the two escalation misses were both edge cases
not captured by the current LLM-based escalation logic. Adding a rule-based
escalation layer — explicit conditions that always trigger escalation regardless
of LLM output (e.g. any query mentioning "ombudsman", "Trading Standards",
"never left the UK and was charged") — would bring escalation accuracy to 100%
on known edge cases.

**3. Customer-facing assistant:** the co-pilot serves agents. The same knowledge
base, RAG pipeline, and Claude integration can power a customer-facing chat widget
on voltex.co.uk — with a different system prompt (friendly, empathetic, no internal
policy language), a clean white interface, and a human handoff mechanism. Industry
data suggests 40–60% of contact centre queries are routine and resolvable without
a human agent. At Voltex's call volume, even a 30% deflection rate represents
millions in annual savings and measurably faster resolution for customers.

---

## Summary

This project demonstrates that a RAG-powered contact centre co-pilot is technically
feasible, economically justified, and practically deployable with a small engineering
team. The architecture — query rewriting, semantic retrieval with category filtering,
structured LLM reasoning, and typed Pydantic output — is production-ready in design
even if not yet in guardrails. The 78.3% evaluation score on genuinely hard retail
policy questions, with 90% retrieval accuracy and 93.3% escalation accuracy, is a
strong foundation to build from.

The gap between 78.3% and production-ready is not a technical gap. It is an
operational one: knowledge base governance, agent training, feedback loops, and
compliance controls. These are solvable problems — and solving them is the work
of an AI Engineer embedded in a real product team.