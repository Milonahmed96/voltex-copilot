"""
Voltex Contact Centre Co-Pilot
Core Reasoning Layer

Architecture:
  Step 1 — Query rewrite:     colloquial customer language → precise retrieval query
  Step 2 — Category detect:   classify which knowledge base category applies
  Step 3 — Retrieval:         top-5 chunks from ChromaDB with category pre-filter
  Step 4 — Reasoning:         Claude generates structured CopilotResponse
  Step 5 — Return:            typed Pydantic output to the UI layer

The query rewrite and category detection are combined into one API call
to minimise latency and cost.
"""

import os
import json
import re
from typing import Literal
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CHROMA_DIR         = Path("chroma_db")
COLLECTION_NAME    = "voltex_knowledge"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
N_RETRIEVAL_CHUNKS = 5

VALID_CATEGORIES = Literal[
    "voltcare",
    "returns_repairs",
    "products",
    "delivery_orders",
    "voltmobile",
    "general",
]

# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

class CopilotResponse(BaseModel):
    """
    The structured output returned for every customer query.
    Designed to be rendered directly in the agent UI.
    """
    customer_need: str = Field(
        description="One sentence summary of what the customer needs, "
                    "written from the agent's perspective."
    )
    suggested_response: str = Field(
        description="The response the agent can send or say to the customer. "
                    "Friendly, clear, under 80 words. No jargon."
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="HIGH if the knowledge base clearly answers the query. "
                    "MEDIUM if the answer is partial or requires agent judgement. "
                    "LOW if the knowledge base does not contain enough information."
    )
    sources_used: list[str] = Field(
        description="List of source document filenames the answer was drawn from."
    )
    key_policy_points: list[str] = Field(
        description="2 to 4 bullet points of the most important policy facts "
                    "the agent should know for this query. For agent reference only — "
                    "not for reading out to the customer."
    )
    escalate: bool = Field(
        description="True if this query requires Team Leader authorisation or "
                    "falls outside standard agent authority. False otherwise."
    )
    escalation_reason: str = Field(
        description="If escalate is True, explain exactly why. "
                    "Empty string if escalate is False."
    )

# ─────────────────────────────────────────────
# DETERMINISTIC ESCALATION RULES
# ─────────────────────────────────────────────

ESCALATION_RULES = [
    # Pattern, reason
    (["ombudsman", "financial ombudsman", "retail ombudsman"],
     "Customer mentioned ombudsman — formal complaint process required"),
    (["trading standards", "trading standard"],
     "Trading Standards reference — compliance team must be involved"),
    (["never left the uk", "didn't leave the uk", "did not leave the uk",
      "stayed in the uk", "roaming charge", "roaming charges"],
     "Roaming charge dispute — check border proximity; TL authorisation for goodwill credit"),
    (["couldn't get through", "couldn't fit", "couldn't get it through",
      "wouldn't fit", "too wide", "too big for the door", "through the door"],
     "Access failure on large item delivery — failed delivery charge waiver assessment required"),
    (["misdescription", "not as described", "wrong description",
      "listing said", "advertised as", "description was wrong"],
     "Potential misdescription — statutory rights apply at any time; escalate to Trading Standards team"),
    (["third party repaired", "neighbour fixed", "friend fixed",
      "someone else fixed", "already been repaired", "repaired before"],
     "Potential unauthorised repair — cannot confirm voidance on call; repair team assessment needed"),
    (["vulnerable", "bereavement", "recently lost", "mental health",
      "dementia", "elderly", "disability", "hardship"],
     "Vulnerable customer indicator — extended cooling-off and payment plan options available"),
    (["fraud", "fraudulent", "scam", "stolen my details",
      "identity theft"],
     "Fraud or identity theft — security team escalation required"),
]


def check_deterministic_escalation(message: str) -> tuple[bool, str]:
    """
    Checks the customer message against known escalation patterns.
    Returns (should_escalate, reason) tuple.
    Runs before LLM reasoning — deterministic and free.
    """
    message_lower = message.lower()
    for patterns, reason in ESCALATION_RULES:
        if any(p in message_lower for p in patterns):
            return True, reason
    return False, ""

# ─────────────────────────────────────────────
# VOLTEX CO-PILOT CLASS
# ─────────────────────────────────────────────

class VoltexCoPilot:
    """
    Main co-pilot class. Initialise once, call get_response() per query.
    Maintains conversation history for multi-turn context.
    """

    def __init__(self):
        print("Initialising Voltex Co-Pilot...")

        # Anthropic client
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Embedding model — same model used during ingestion
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # ChromaDB connection
        print(f"  Connecting to ChromaDB at: {CHROMA_DIR.resolve()}")
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = chroma_client.get_collection(COLLECTION_NAME)
        chunk_count = self.collection.count()
        print(f"  Collection '{COLLECTION_NAME}' loaded — {chunk_count} chunks")

        # Conversation history for multi-turn context
        self.conversation_history = []
        self.MAX_HISTORY_TURNS = 4

        # Cross-encoder re-ranker — scores top-N chunks for precise relevance
        print(f"  Loading cross-encoder re-ranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("  Co-Pilot ready.\n")


    # ─────────────────────────────────────────────
    # STEP 1 + 2 — QUERY REWRITE AND CATEGORY DETECT
    # ─────────────────────────────────────────────

    def _rewrite_and_classify(
        self,
        customer_message: str,
        conversation_history: list[dict],
    ) -> tuple[str, str]:
        """
        Combines query rewriting and category classification into one API call.

        Returns:
            rewritten_query: optimised for vector similarity search
            category: one of the VALID_CATEGORIES strings
        """

        # Build context from recent conversation turns if available
        history_context = ""
        if conversation_history:
            recent = conversation_history[-2:]  # last 1-2 turns
            history_context = "\n".join([
                f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content'][:200]}"
                for m in recent
            ])
            history_context = f"\nRecent conversation context:\n{history_context}\n"

        prompt = f"""You are helping a contact centre agent at Voltex, a UK technology and appliances retailer.

        A customer has just said the following:
        "{customer_message}"
        {history_context}
        Your job is to output a JSON object with exactly two fields:

        1. "rewritten_query": Rewrite the customer's message as a precise search query for a retail knowledge base.
        Use specific retail and policy terminology. Replace ALL vague or colloquial terms with precise ones.

        CRITICAL TRANSLATION EXAMPLES — always apply these patterns:
        - "battery drains fast / dies quickly / doesn't last" → "laptop battery degradation premature failure warranty coverage charge cycles"
        - "phone won't turn on / died / stopped working" → "phone hardware failure repair warranty claim"
        - "going to [any country]" → identify if EU/EEA or not, then: "EU roaming VoltMobile phone usage abroad" OR "Zone A/B/C roaming VoltMobile abroad"
        - "charged for roaming / roaming bill" + "didn't leave UK" → "border proximity roaming charge dispute goodwill credit"
        - "fridge/washer/TV didn't arrive / not delivered" → "large item delivery missed failed delivery [product type]"
        - "cancel contract / leave VoltMobile" → "contract cancellation early termination VoltMobile SIM plan"
        - "stop paying / not pay" → "contract non-payment default consequences debt collection"
        - "screen has lines / display fault / TV broken" → "screen fault manufacturing defect warranty statutory rights"
        - "couldn't get it through the door / doesn't fit" → "large item delivery access failure failed delivery doorway width"
        - "neighbour / friend / third party fixed it" → "unauthorised repair third party voided VoltCare cover"
        - "bought it as a gift / giving as present" → "gift purchase return refund proof of purchase"
        - "product listing said / advertised as / description wrong" → "product misdescription not as described statutory rights Trading Standards"
        - "been with you for years / long-standing customer" → "customer retention goodwill discretionary"
        - "ombudsman / complain / escalate" → "formal complaint escalation Financial Ombudsman"
        - "how do I claim / make a claim / start a claim" → "VoltCare claim process steps VC- plan reference voltex.co.uk online phone store"

        2. "category": The single most relevant knowledge base category. Must be exactly one of:
        - "voltcare" — warranty, care plans, VoltCare claims, repair cover
        - "returns_repairs" — returns, refunds, exchanges, repairs, Consumer Rights Act
        - "products" — product specifications, buying advice, technical questions
        - "delivery_orders" — delivery, tracking, click and collect, VoltInstall, order amendments
        - "voltmobile" — VoltMobile SIM plans, phone contracts, roaming, billing, porting
        - "general" — only if the query genuinely spans multiple categories equally

        CATEGORY DECISION RULES:
        - Any mention of roaming, SIM, phone plan, mobile data, PAC code → "voltmobile"
        - Any mention of delivery, arrived, didn't come, tracking, install → "delivery_orders"
        - Any mention of VoltCare, warranty, claim, repair cover → "voltcare"
        - Any mention of return, refund, broken, fault, Consumer Rights → "returns_repairs"
        - Any mention of which TV/laptop/washer to buy, specs, difference between → "products"
        - Battery questions: if about laptop battery life/performance → "products"; if about warranty coverage for battery → "voltcare"

        Respond with ONLY the JSON object. No explanation, no markdown, no preamble.
        Example: {{"rewritten_query": "EU roaming VoltMobile phone usage abroad", "category": "voltmobile"}}"""

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Parse JSON — strip any accidental markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            parsed = json.loads(raw)
            rewritten_query = parsed.get("rewritten_query", customer_message)
            category = parsed.get("category", "general")

            # Validate category
            valid = {
                "voltcare", "returns_repairs", "products",
                "delivery_orders", "voltmobile", "general"
            }
            if category not in valid:
                category = "general"

        except json.JSONDecodeError:
            # If parsing fails, use the original query and no filter
            rewritten_query = customer_message
            category = "general"

        return rewritten_query, category


    # ─────────────────────────────────────────────
    # STEP 3 — RETRIEVAL
    # ─────────────────────────────────────────────

    def _retrieve(
        self,
        rewritten_query: str,
        category: str,
        n_results: int = N_RETRIEVAL_CHUNKS,
    ) -> list[dict]:
        """
        Embeds the rewritten query and retrieves top-N chunks from ChromaDB.
        Applies category pre-filter unless category is 'general'.

        Returns list of dicts with 'text', 'source', 'similarity', 'section'.
        """
        query_embedding = self.model.encode(rewritten_query).tolist()

        # Build query args
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        # Apply category filter for non-general queries
        if category != "general":
            query_kwargs["where"] = {"category": category}

        results = self.collection.query(**query_kwargs)

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            chunks.append({
                "text"      : doc,
                "source"    : meta.get("source", "unknown"),
                "category"  : meta.get("category", "unknown"),
                "section"   : meta.get("section", ""),
                "similarity": round(1 - dist, 3),
            })

        # Re-rank with cross-encoder for precise relevance scoring
        if chunks:
            pairs = [(rewritten_query, chunk["text"]) for chunk in chunks]
            rerank_scores = self.reranker.predict(pairs).tolist()
            for chunk, score in zip(chunks, rerank_scores):
                chunk["rerank_score"] = round(float(score), 3)
            chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return chunks


    # ─────────────────────────────────────────────
    # STEP 4 — REASONING
    # ─────────────────────────────────────────────

    def _reason(
        self,
        customer_message: str,
        rewritten_query: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict],
    ) -> CopilotResponse:
        """
        Sends the retrieved context and conversation history to Claude.
        Returns a structured CopilotResponse.
        """

        # Format retrieved chunks for the prompt
        context_block = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_block += (
                f"\n[Source {i}: {chunk['source']} | "
                f"similarity: {chunk['similarity']}]\n"
                f"{chunk['text']}\n"
                f"{'─' * 40}"
            )

        # System prompt — this is the most carefully designed part
        system_prompt = """You are a real-time assistant for Voltex contact centre agents.
Voltex is a UK omnichannel technology and appliances retailer offering products, VoltCare protection plans, VoltMobile SIM and phone plans, and VoltInstall installation services.

Your role is to help agents respond to customers accurately and quickly during live calls or chats.

RULES YOU MUST FOLLOW:
1. Base your answer ONLY on the knowledge base excerpts provided. Never invent policy details, prices, timescales, or phone numbers.
2. If the knowledge base does not contain enough information to answer confidently, say so clearly — set confidence to LOW and escalate if appropriate.
3. The suggested_response field is what the agent will say or send to the customer — write it in friendly, plain English, under 80 words, no internal jargon.
4. The key_policy_points field is for the agent's reference only — include precise policy details the agent needs to know (timescales, thresholds, conditions) that would be too detailed to say to the customer directly.
5. Set escalate to True when: the query requires Team Leader authorisation, the situation is outside standard agent authority, the customer is vulnerable, or there is a potential compliance issue (mis-sale, Consumer Rights Act dispute, porting error).
6. Never suggest the customer has caused a fault without evidence. The burden of proof in the first 6 months is on Voltex.
7. When a query involves both VoltCare and statutory rights, always mention both options if both are relevant.

OUTPUT FORMAT:
Respond with a single valid JSON object matching this exact schema:
{
  "customer_need": "one sentence from the agent's perspective",
  "suggested_response": "what the agent says to the customer, under 80 words",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "sources_used": ["filename1.txt", "filename2.txt"],
  "key_policy_points": ["point 1", "point 2", "point 3"],
  "escalate": true | false,
  "escalation_reason": "reason if escalate is true, empty string if false"
}

Respond with ONLY the JSON. No markdown fences, no explanation."""

        # Build messages — include conversation history for context
        messages = []

        # Add recent conversation history (last MAX_HISTORY_TURNS turns)
        if conversation_history:
            messages.extend(conversation_history[-(self.MAX_HISTORY_TURNS * 2):])

        # Add the current query with retrieved context
        user_message = f"""Customer message: "{customer_message}"

Rewritten search query used: "{rewritten_query}"

Knowledge base excerpts retrieved:
{context_block}

Based on the knowledge base excerpts above, generate the CopilotResponse JSON for this customer query."""

        messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            parsed = json.loads(raw)
            return CopilotResponse(**parsed)

        except (json.JSONDecodeError, Exception) as e:
            # Graceful fallback — return a LOW confidence response
            return CopilotResponse(
                customer_need="Unable to parse response — please handle manually.",
                suggested_response="I'd like to help you with that. Could you give me "
                                   "a moment to check the details for you?",
                confidence="LOW",
                sources_used=[],
                key_policy_points=[f"System error: {str(e)[:100]}"],
                escalate=True,
                escalation_reason="System parsing error — handle manually.",
            )


    # ─────────────────────────────────────────────
    # STEP 5 — PUBLIC INTERFACE
    # ─────────────────────────────────────────────

    def get_response(self, customer_message: str) -> dict:
        """
        Main public method. Takes a customer message string.
        Returns a dict containing the CopilotResponse plus debug metadata.

        The debug metadata (rewritten_query, category, retrieved_chunks)
        is used by the Streamlit UI to show the agent what happened
        under the hood — important for trust and transparency.
        """

        # Step 1+2: rewrite and classify
        rewritten_query, category = self._rewrite_and_classify(
            customer_message,
            self.conversation_history,
        )

        # Step 3: retrieve
        retrieved_chunks = self._retrieve(rewritten_query, category)

        # Step 3.5: deterministic escalation check
        rule_escalate, rule_reason = check_deterministic_escalation(customer_message)

        # Step 4: reason
        copilot_response = self._reason(
            customer_message,
            rewritten_query,
            retrieved_chunks,
            self.conversation_history,
        )

        # Step 4.5: override escalation with deterministic rules if triggered
        if rule_escalate and not copilot_response.escalate:
            copilot_response = copilot_response.model_copy(update={
                "escalate": True,
                "escalation_reason": rule_reason,
            })

        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": customer_message,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": copilot_response.suggested_response,
        })

        # Return everything the UI needs
        return {
            "response"        : copilot_response,
            "rewritten_query" : rewritten_query,
            "category"        : category,
            "retrieved_chunks": retrieved_chunks,
        }

    def reset_conversation(self):
        """Clear conversation history — call at start of each new customer interaction."""
        self.conversation_history = []


# ─────────────────────────────────────────────
# COMMAND LINE TEST HARNESS
# ─────────────────────────────────────────────

def print_response(result: dict):
    """Pretty-prints a CopilotResponse to the terminal."""
    r = result["response"]

    confidence_icon = {"HIGH": "✓", "MEDIUM": "~", "LOW": "✗"}.get(r.confidence, "?")

    print("\n" + "=" * 60)
    print(f"QUERY REWRITTEN AS : {result['rewritten_query']}")
    print(f"CATEGORY DETECTED  : {result['category']}")
    print(f"CONFIDENCE         : {confidence_icon} {r.confidence}")
    print(f"ESCALATE           : {'YES — ' + r.escalation_reason if r.escalate else 'No'}")
    print("-" * 60)
    print(f"CUSTOMER NEED:\n  {r.customer_need}")
    print(f"\nSUGGESTED RESPONSE:\n  {r.suggested_response}")
    print(f"\nKEY POLICY POINTS FOR AGENT:")
    for point in r.key_policy_points:
        print(f"  • {point}")
    print(f"\nSOURCES USED: {', '.join(r.sources_used) if r.sources_used else 'none'}")
    print("\nTOP RETRIEVED CHUNKS:")
    for i, chunk in enumerate(result["retrieved_chunks"][:3], 1):
        print(f"  [{i}] sim={chunk['similarity']} | {chunk['source']}")
        print(f"      {chunk['text'][:120].replace(chr(10), ' ')}...")
    print("=" * 60)


if __name__ == "__main__":
    copilot = VoltexCoPilot()

    # Test queries — one per category, including multi-turn test
    test_queries = [
        "Does VoltCare cover if I accidentally drop my laptop?",
        "I bought a TV three weeks ago and the screen has developed a fault",
        "What is the difference between OLED and QLED?",
        "My washing machine was supposed to arrive yesterday but it hasn't come",
        "I am going on holiday to Spain next week — will my phone work?",
    ]

    print("\nRunning test queries...\n")

    for query in test_queries:
        print(f"\n>>> Customer: \"{query}\"")
        result = copilot.get_response(query)
        print_response(result)

    # Multi-turn test — conversation with follow-up questions
    print("\n" + "=" * 60)
    print("MULTI-TURN CONVERSATION TEST")
    print("=" * 60)

    copilot.reset_conversation()

    conversation = [
        "I want to cancel my VoltMobile contract",
        "I still have 8 months left on my contract",
        "What if I just stop paying?",
    ]

    for turn in conversation:
        print(f"\n>>> Customer: \"{turn}\"")
        result = copilot.get_response(turn)
        print_response(result)