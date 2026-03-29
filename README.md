# ⚡ Voltex Co-Pilot

> AI-powered contact centre assistant for Voltex Retail — a fictional UK omnichannel technology and appliances retailer built as an AI engineering simulation project.

[![CI](https://github.com/Milonahmed96/voltex-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Milonahmed96/voltex-copilot/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/live%20demo-streamlit-FF4B4B)](https://voltex-copilot-ssckdkfbuq3diathsc66z8.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Evaluation](https://img.shields.io/badge/evaluation-82.0%25-brightgreen)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)

## Live Demo

🚀 **[Try Voltex Co-Pilot here](https://voltex-copilot-ssckdkfbuq3diathsc66z8.streamlit.app/)**


---

## What This Is

Voltex Co-Pilot is a RAG-powered contact centre assistant that helps agents respond to customer queries accurately and instantly during live calls. It searches across a 23,000-word knowledge base of retail policy documents in real time, rewrites colloquial customer language into precise retrieval queries, and generates structured responses grounded in policy — never invented.

This project simulates the kind of AI tooling being deployed at UK tech retailers like Currys, where contact centre agents handle millions of queries per year across warranty claims, returns, product advice, delivery, and mobile plan support. The goal was to build genuine business understanding of the problem, not just a demo.

**Voltex Retail is a fictional business** created specifically for this project. All policies, products, and business data are synthetic. The architecture and evaluation methodology reflect real-world production RAG patterns.

---

## Business Problem

A Voltex contact centre agent handles 80–100 calls per day across five knowledge domains:

- VoltCare warranty and care plans
- Returns, repairs, and Consumer Rights Act obligations
- Product specifications and buying advice
- Delivery, installation, and order management
- VoltMobile SIM and phone plan support

Each domain has policy documents running to thousands of words with version histories, edge cases, and interaction effects between categories. Agents who don't know the exact policy at the moment of the call make errors that generate complaints, compensation costs, and regulatory risk. Common failure modes include applying incorrect excess charges to legacy plan holders, denying claims during the Grace Period when cover legally continues, and quoting only the service early termination charge on phone cancellations without mentioning the handset settlement balance.

The co-pilot addresses this by surfacing the exact relevant policy at the moment of the call and generating a suggested response grounded in that policy.

---

## Architecture

```
Customer message
      │
      ▼
┌─────────────────────────────┐
│  Step 1: Query Rewrite      │  Colloquial language → precise retrieval terms
│  + Category Classification  │  LLM-based, one API call
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 2: Deterministic      │  Rule-based escalation triggers
│  Escalation Check           │  Free, runs before any LLM call
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 3: Retrieval          │  ChromaDB vector search with category filter
│                             │  Cross-encoder re-ranking for precision
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Step 4: Reasoning          │  Claude Sonnet with structured system prompt
│                             │  Conversation history for multi-turn context
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  CopilotResponse (Pydantic) │  customer_need, suggested_response,
│                             │  confidence, sources, key_policy_points,
│                             │  escalate, escalation_reason
└─────────────────────────────┘
```

### Key Design Decisions

**Dual chunking strategy** — policy documents use a sliding window chunker (600 chars, 150 overlap) to preserve cross-sentence policy context. The FAQ document uses Q&A boundary chunking to keep each question and answer as one atomic retrieval unit. This distinction produced measurably better results than a single chunking strategy.

**Query rewriting before retrieval** — customer language and policy language are semantically distant. "Going to Spain next week" fails to retrieve EU roaming policy. After rewriting to "EU roaming VoltMobile phone usage abroad Spain", similarity jumps from 0.407 to 0.709. A dedicated rewrite step is not optional for retail contact centre use cases.

**Deterministic escalation layer** — an LLM-based escalation flag misses edge cases. A rule-based pre-check for known patterns (ombudsman mentions, roaming charge disputes, misdescription claims, vulnerable customer indicators) catches these deterministically before the LLM reasoning step.

**Cross-encoder re-ranking** — bi-encoder retrieval is fast but approximate. A cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`) scores the top-N retrieved chunks against the query for precise relevance ordering, improving hard question performance by 7.4 percentage points.

---

## Knowledge Base

Five synthetic policy documents totalling 23,000 words across 305 indexed chunks:

| Document | Words | Chunks | Domain |
|---|---|---|---|
| `voltcare_policy.txt` | 4,006 | 61 | Warranty and care plans |
| `repairs_returns_policy.txt` | 4,013 | 61 | Returns, refunds, Consumer Rights Act |
| `product_faqs.txt` | 5,427 | 35 | Product specifications and buying advice |
| `delivery_orders_policy.txt` | 4,662 | 71 | Delivery, installation, order management |
| `voltmobile_policy.txt` | 5,064 | 77 | Mobile plans, roaming, billing, porting |

Documents are written as dual-reader design — plain English accessible to customers, structured for rapid agent navigation. Each document includes version history, edge cases, agent-specific decision guidance, and a quick reference section.

---

## Evaluation Results

Evaluated across 50 questions in two independent test sets:

### Golden Set (30 questions — used during optimisation)

| Metric | Score |
|---|---|
| Overall | 81.1% (146/180) |
| Retrieval accuracy | 93.3% |
| Easy questions | 86.7% |
| Medium questions | 80.3% |
| Hard questions | 75.9% |
| Escalation accuracy | 96.7% |

### Independent Test Set (20 questions — unseen during optimisation)

| Metric | Score |
|---|---|
| Overall | 83.3% (100/120) |
| Retrieval accuracy | 90.0% |
| Products category | 100% |
| Cross-category queries | 94.4% |
| Escalation accuracy | 95.0% |

**System generalises** — the independent test set scored 2.2 percentage points *higher* than the golden set, confirming the optimisations improved genuine understanding rather than overfitting to specific phrasings.

### Scoring Methodology

Each question is scored across four dimensions (maximum 6 points):
- **Retrieval (0/1)** — correct source document in top 3 chunks
- **Response quality (1–3)** — essential policy keyword coverage
- **Confidence appropriateness (0/1)** — HIGH/MEDIUM/LOW correctly calibrated
- **Escalation accuracy (0/1)** — escalation flag matches expected behaviour

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic API) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector database | ChromaDB (persistent local) |
| Output schema | Pydantic v2 |
| UI | Streamlit |
| Testing | pytest (33 tests) |
| CI | GitHub Actions |
| Python | 3.11 |

---

## Project Structure

```
voltex-copilot/
├── knowledge_base/                  # 5 synthetic policy documents
│   ├── voltcare_policy.txt
│   ├── repairs_returns_policy.txt
│   ├── product_faqs.txt
│   ├── delivery_orders_policy.txt
│   └── voltmobile_policy.txt
├── chroma_db/                       # Persisted vector index (gitignored)
├── evaluation/                      # Evaluation outputs
│   ├── eval_results.json            # Golden set results
│   ├── eval_results_v2.json         # Independent set results
│   ├── eval_report.md               # Golden set markdown report
│   └── ingestion_summary.json
├── tests/                           # 33 pytest tests
│   ├── test_chunking.py             # 11 chunking logic tests
│   ├── test_schema.py               # 10 Pydantic schema tests
│   └── test_copilot_unit.py         # 12 unit tests with mocks
├── .github/workflows/ci.yml         # GitHub Actions CI
├── ingest.py                        # Knowledge base ingestion pipeline
├── copilot.py                       # Core reasoning layer
├── app.py                           # Streamlit UI
├── evaluate.py                      # Golden set evaluation (30Q)
├── evaluate_v2.py                   # Independent evaluation (20Q)
├── business_reflection.md           # Business analysis document
├── requirements.txt
└── .env                             # API keys (gitignored)
```

---

## Getting Started

### Prerequisites

- Python 3.11
- Anthropic API key (get one at console.anthropic.com)

### Installation

```bash
git clone https://github.com/Milonahmed96/voltex-copilot.git
cd voltex-copilot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

### Build the Knowledge Base

```bash
python ingest.py
```

This chunks and embeds all 5 policy documents into ChromaDB. Takes approximately 60 seconds on first run (downloads the embedding model). Subsequent runs take under 10 seconds.

### Run the UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The UI includes 6 quick test queries — click any to see the co-pilot in action.

### Run Evaluation

```bash
python evaluate.py      # 30-question golden set (~3 min, uses API)
python evaluate_v2.py   # 20-question independent set (~2 min, uses API)
```

### Run Tests

```bash
python -m pytest tests/ -v
```

All 33 tests pass without an API key — external dependencies are fully mocked.

---

## Business Context

This project was built as part of a simulation to develop genuine business understanding of AI engineering in UK retail. The three projects in this series are:

| Project | Repo | Description |
|---|---|---|
| B — Contact Centre Co-Pilot | [voltex-copilot](https://github.com/Milonahmed96/voltex-copilot) · [Live demo](https://voltex-copilot-ssckdkfbuq3diathsc66z8.streamlit.app/) | RAG-powered agent assistant, 82% evaluation accuracy |
| A — ShopFloor Analyst | [voltex-shopfloor](https://github.com/Milonahmed96/voltex-shopfloor) · [Live demo](https://voltex-shopfloor-ma.streamlit.app/) | LLM store operations reasoning, 37 tests |
| C — Repair Triage Agent | coming soon | LangGraph agentic repair routing |

The fictional retailer Voltex is modelled on the operational structure of UK omnichannel tech retailers, with realistic policy complexity, service architecture (MVNO, extended warranty, in-house repair centre), and contact centre scale.

---

## Author

**Milon Ahmed**
MSc Data Science with Advanced Research, University of Hertfordshire (graduating October 2026)
BSc Mathematics

[GitHub](https://github.com/Milonahmed96) · [LinkedIn](https://linkedin.com/in/milonahmed96)

---

## Licence

MIT — see LICENSE file.
