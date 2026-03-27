"""
Tests for the chunking functions in ingest.py.
These run without any external dependencies — no API calls, no ChromaDB.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import chunk_policy_document, chunk_faq_document


# ─────────────────────────────────────────────
# POLICY CHUNKING TESTS
# ─────────────────────────────────────────────

def test_policy_chunker_returns_list():
    text = "This is a sentence. This is another sentence. And a third one here."
    chunks = chunk_policy_document(text, chunk_size=400, overlap=80)
    assert isinstance(chunks, list)


def test_policy_chunker_produces_chunks_from_real_text():
    text = """
    VoltCare Essential covers mechanical and electrical breakdown only.
    It does not cover accidental damage such as drops or spills.
    The plan is available on all product categories sold by Voltex.
    Duration options are one or two years selected at point of purchase.
    Monthly pricing starts from three pounds ninety nine per month.
    A ten percent discount applies when paying the annual amount upfront.
    Customers may cancel within forty five days for a full refund.
    After forty five days a pro rata refund applies minus fifteen pounds admin fee.
    """ * 5  # repeat to ensure multiple chunks are produced
    chunks = chunk_policy_document(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1, "Expected multiple chunks from long text"


def test_policy_chunker_respects_chunk_size():
    # Text with sentence boundaries so the chunker can split properly
    sentence = "This is a valid sentence with enough words to test chunking. "
    text = sentence * 40  # ~2400 characters with sentence boundaries
    chunks = chunk_policy_document(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1, "Expected multiple chunks"
    # Allow 2x headroom because chunker splits on sentences not hard character cuts
    for chunk in chunks:
        assert len(chunk) < 200 * 2, f"Chunk too long: {len(chunk)} chars"


def test_policy_chunker_overlap_provides_context():
    # With overlap, consecutive chunks should share some content
    text = (
        "The Grace Period lasts fourteen days. "
        "Cover continues during the Grace Period. "
        "Payment must be resolved before the period ends. "
        "If unresolved the plan enters suspension. "
        "Suspension means cover stops immediately. "
    ) * 10
    chunks = chunk_policy_document(text, chunk_size=150, overlap=50)
    assert len(chunks) >= 2
    # The last words of chunk N should appear somewhere in chunk N+1
    if len(chunks) >= 2:
        last_words_of_first = chunks[0].split()[-3:]
        second_chunk_words  = chunks[1].split()
        overlap_found = any(w in second_chunk_words for w in last_words_of_first)
        assert overlap_found, "Expected overlap between consecutive chunks"


def test_policy_chunker_filters_very_short_chunks():
    # Short divider lines and headers should be filtered out
    text = "\n".join([
        "━" * 50,
        "PART 1",
        "This is a proper sentence with enough content to be a valid chunk worth keeping.",
        "━" * 50,
        "PART 2",
        "Another proper sentence that contains meaningful policy information for retrieval.",
    ])
    chunks = chunk_policy_document(text, chunk_size=400, overlap=80)
    for chunk in chunks:
        assert len(chunk) > 40, f"Short chunk not filtered: '{chunk}'"


# ─────────────────────────────────────────────
# FAQ CHUNKING TESTS
# ─────────────────────────────────────────────

def test_faq_chunker_returns_list():
    text = "Q: What is VoltCare?\nA: VoltCare is Voltex's protection plan."
    chunks = chunk_faq_document(text)
    assert isinstance(chunks, list)


def test_faq_chunker_splits_on_question_boundaries():
    text = """
Q: What does VoltCare Essential cover?
A: It covers mechanical and electrical breakdown only.
It does not cover accidental damage.

Q: What does VoltCare Plus cover?
A: It covers everything in Essential plus accidental damage.
One accidental damage claim is allowed per rolling twelve month period.

Q: What does VoltCare Complete cover?
A: It covers everything in Plus with unlimited accidental damage claims.
No excess applies on any claim under Complete.
"""
    chunks = chunk_faq_document(text)
    assert len(chunks) == 3, f"Expected 3 Q&A chunks, got {len(chunks)}"


def test_faq_chunker_keeps_question_and_answer_together():
    text = """
Q: Does VoltCare cover accidental damage on a laptop?
A: Yes, accidental damage is covered under VoltCare Plus and Complete.
An excess applies on Plus claims depending on the product value.
Complete has no excess on any claim.
"""
    chunks = chunk_faq_document(text)
    assert len(chunks) == 1
    assert "Q:" in chunks[0]
    assert "A:" in chunks[0]
    assert "accidental damage" in chunks[0]


def test_faq_chunker_prepends_section_header():
    text = """
SECTION B — TELEVISIONS

Q: What is the difference between OLED and QLED?
A: OLED pixels produce their own light. QLED uses quantum dot enhancement over LED backlight.
"""
    chunks = chunk_faq_document(text)
    assert len(chunks) == 1
    assert "SECTION B" in chunks[0], "Section header should be prepended to chunk"


def test_faq_chunker_handles_multiple_sections():
    text = """
SECTION A — LAPTOPS AND COMPUTING

Q: How much RAM do I need?
A: For everyday use sixteen gigabytes is the recommended minimum in 2026.

SECTION B — TELEVISIONS

Q: What screen size should I buy?
A: Match screen size to your viewing distance. Fifty five inches suits most living rooms.
"""
    chunks = chunk_faq_document(text)
    assert len(chunks) == 2
    assert "SECTION A" in chunks[0]
    assert "SECTION B" in chunks[1]


def test_faq_chunker_filters_very_short_chunks():
    text = """
SECTION A — LAPTOPS

Q: What is RAM?
A: RAM is memory. It determines how many things your laptop can handle at once.

Q: Hi
A: No.
"""
    chunks = chunk_faq_document(text)
    # The "Hi / No" chunk is too short and should be filtered
    for chunk in chunks:
        assert len(chunk) > 60, f"Short chunk not filtered: '{chunk}'"