"""
Voltex Contact Centre Co-Pilot
Streamlit UI v2
"""

import streamlit as st
from copilot import VoltexCoPilot

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Voltex Co-Pilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Premium dark background */
    .stApp {
        background: #0F1923;
    }

    /* Main container */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }

    /* Full-width hero header */
    .copilot-hero {
        background: linear-gradient(135deg, #0F6E56 0%, #1D9E75 50%, #0F6E56 100%);
        color: white;
        padding: 2.5rem 3rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #5DCAA5;
    }
    .copilot-hero h1 {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        color: white;
        letter-spacing: -0.5px;
    }
    .copilot-hero .subtitle {
        font-size: 1.05rem;
        opacity: 0.88;
        margin: 0 0 1.25rem 0;
        color: #E1F5EE;
    }
    .copilot-hero .intro {
        font-size: 0.92rem;
        opacity: 0.80;
        line-height: 1.7;
        max-width: 780px;
        color: #E1F5EE;
        border-top: 1px solid rgba(255,255,255,0.2);
        padding-top: 1rem;
        margin-top: 0.5rem;
    }
    .copilot-hero .badge-row {
        display: flex;
        gap: 10px;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .copilot-hero .hero-badge {
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.25);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Section labels */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #5DCAA5;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }

    /* Card wrapper for panels */
    .panel-card {
        background: #1A2535;
        border: 1px solid #253347;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* Suggested response box */
    .response-box {
        background: #0D3D2E;
        border-left: 4px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #9FE1CB;
    }

    /* Escalation banner */
    .escalate-banner {
        background: #2D1515;
        border-left: 4px solid #E24B4A;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.9rem;
        color: #F7C1C1;
        font-weight: 500;
    }

    /* Confidence badges */
    .badge-high   { background:#0D3D2E; color:#5DCAA5; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; border:1px solid #1D9E75; }
    .badge-medium { background:#2D1E0A; color:#FAC775; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; border:1px solid #BA7517; }
    .badge-low    { background:#2D1515; color:#F7C1C1; padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; border:1px solid #E24B4A; }

    /* Category pill */
    .category-pill { background:#1A1A3A; color:#AFA9EC; padding:4px 12px; border-radius:20px; font-size:0.8rem; border:1px solid #534AB7; }

    /* Policy points */
    .policy-point {
        padding: 0.45rem 0;
        border-bottom: 1px solid #253347;
        font-size: 0.87rem;
        color: #B4B2A9;
        line-height: 1.5;
    }
    .policy-point:last-child { border-bottom: none; }

    /* Turn counter */
    .turn-counter {
        font-size: 0.78rem;
        color: #5DCAA5;
        text-align: right;
        margin-top: 0.25rem;
        opacity: 0.8;
    }

    /* Quick query buttons styling override */
    .quick-btn {
        background: #1A2535;
        border: 1px solid #253347;
        color: #B4B2A9;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.82rem;
        text-align: left;
        width: 100%;
        margin-bottom: 6px;
        cursor: pointer;
        transition: all 0.15s;
    }

    /* Chunk similarity bars */
    .chunk-card {
        background: #141E2B;
        border-radius: 8px;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1D9E75;
    }
    .chunk-card.med { border-left-color: #BA7517; }
    .chunk-card.low { border-left-color: #E24B4A; }

    /* Text area and inputs */
    .stTextArea textarea {
        background: #141E2B !important;
        color: #D3D1C7 !important;
        border: 1px solid #253347 !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #1D9E75 !important;
        box-shadow: 0 0 0 2px rgba(29,158,117,0.2) !important;
    }

    /* Placeholder text */
    .stTextArea textarea::placeholder { color: #4A4A4A !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s !important;
    }
    .stButton > button[kind="primary"] {
        background: #1D9E75 !important;
        border: none !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 0.6rem 1rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #0F6E56 !important;
    }
    .stButton > button:not([kind="primary"]) {
        background: #1A2535 !important;
        border: 1px solid #253347 !important;
        color: #B4B2A9 !important;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #1D9E75 !important;
        color: #5DCAA5 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #141E2B !important;
        color: #888 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
    }

    /* Rewritten query box */
    .rewrite-box {
        background: #141E2B;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.82rem;
        color: #5DCAA5;
        font-family: monospace;
        margin-top: 0.25rem;
        border: 1px solid #253347;
    }

    /* Source code tags */
    code {
        background: #141E2B !important;
        color: #5DCAA5 !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-size: 0.78rem !important;
    }

    /* General text */
    p, li, div { color: #D3D1C7; }
    h1, h2, h3 { color: #E1F5EE; }

    /* Spinner */
    .stSpinner > div { border-top-color: #1D9E75 !important; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }

    /* Divider */
    hr { border-color: #253347 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INITIALISE CO-PILOT
# ─────────────────────────────────────────────

@st.cache_resource
def load_copilot():
    return VoltexCoPilot()

copilot = load_copilot()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

if "turn_count"   not in st.session_state: st.session_state.turn_count   = 0
if "last_result"  not in st.session_state: st.session_state.last_result  = None
if "last_query"   not in st.session_state: st.session_state.last_query   = ""
if "input_text"   not in st.session_state: st.session_state.input_text   = ""

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="copilot-hero">
    <h1>⚡ Voltex Co-Pilot</h1>
    <p class="subtitle">Contact Centre Assistant &nbsp;·&nbsp; Internal Use Only</p>
    <p class="intro">
        Voltex Co-Pilot is an AI-powered assistant built for Voltex contact centre agents.
        It listens to what a customer says, searches across Voltex's knowledge base in real time,
        and suggests the best response — including policy details, escalation guidance, and
        key points to know before you reply. Co-Pilot covers VoltCare warranty claims,
        returns and repairs, product advice, delivery queries, and VoltMobile plan support.
        Every response is grounded in Voltex policy — the AI never invents information.
    </p>
    <div class="badge-row">
        <span class="hero-badge">⚡ 421 knowledge chunks</span>
        <span class="hero-badge">📄 5 policy documents</span>
        <span class="hero-badge">🤖 Claude Sonnet</span>
        <span class="hero-badge">🔍 Semantic retrieval</span>
        <span class="hero-badge">🔄 Multi-turn context</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────

if st.session_state.last_result is None:
    left, right = st.columns([1.15, 1], gap="large")
else:
    left, right = st.columns([1, 1.8], gap="large")

# ─────────────────────────────────────────────
# LEFT COLUMN — chat input
# ─────────────────────────────────────────────

with left:
    st.markdown('<div class="section-label">What did the customer just say?</div>', unsafe_allow_html=True)

    customer_message = st.text_area(
        label="Customer message",
        placeholder="Type or paste what the customer just said...\n\ne.g. My washing machine was delivered yesterday but it's making a very loud noise when spinning and I'm worried it's broken.",
        height=180,
        label_visibility="collapsed",
        value=st.session_state.input_text,
    )

    col_submit, col_reset = st.columns([2, 1])
    with col_submit:
        submit = st.button("Get response ↗", type="primary", use_container_width=True)
    with col_reset:
        reset = st.button("New customer", use_container_width=True)

    if reset:
        copilot.reset_conversation()
        st.session_state.turn_count = 0
        st.session_state.last_result = None
        st.session_state.last_query = ""
        st.session_state.input_text = ""
        st.rerun()

    if st.session_state.turn_count > 0:
        st.markdown(
            f'<div class="turn-counter">Turn {st.session_state.turn_count} · conversation active</div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.last_result is not None:
        st.markdown(
            '<div class="turn-counter">Turn 1 · conversation active</div>',
            unsafe_allow_html=True,
        )

    # ── How to use
    st.markdown('<div class="section-label" style="margin-top:1.75rem;">How to use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#888; line-height:1.9;">
    1. Type what the customer just said in the box above<br>
    2. Click <strong style="color:#5DCAA5;">Get response</strong><br>
    3. Read the suggested response (green box on right)<br>
    4. Check key policy points before replying<br>
    5. Escalate if the red banner appears<br>
    6. Click <strong style="color:#5DCAA5;">New customer</strong> to start a fresh conversation
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RIGHT COLUMN — quick queries + response
# ─────────────────────────────────────────────

with right:

    # ── Quick test queries at top of right column
    st.markdown('<div class="section-label">Quick test queries</div>', unsafe_allow_html=True)

    test_queries = [
        "Does VoltCare cover accidental damage on my laptop?",
        "My TV has developed a fault 3 weeks after buying it",
        "I'm going to Spain next week, will my phone work?",
        "My washing machine hasn't arrived and it was due yesterday",
        "I want to cancel my VoltMobile contract",
        "What's the difference between OLED and QLED?",
    ]

    cols = st.columns(2)
    for i, q in enumerate(test_queries):
        with cols[i % 2]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                with st.spinner("Retrieving and reasoning..."):
                    result = copilot.get_response(q)
                    st.session_state.last_result = result
                    st.session_state.last_query = q
                    st.session_state.turn_count += 1
                st.rerun()

    st.divider()

    # ── Process submitted query
    if submit and customer_message.strip():
        with st.spinner("Retrieving and reasoning..."):
            result = copilot.get_response(customer_message.strip())
            st.session_state.last_result = result
            st.session_state.last_query = customer_message.strip()
            st.session_state.turn_count += 1
        st.rerun()

    # ── Response display
    result = st.session_state.last_result

    if result is None:
        st.markdown("""
        <div style="color:#3A4A5A; font-size:0.92rem; margin-top:1.5rem;
                    text-align:center; line-height:2.2; padding:2rem;
                    border:1px dashed #253347; border-radius:12px;">
            ⚡ Co-Pilot is ready<br>
            <span style="font-size:0.82rem;">
                Type a customer query on the left or click a quick test above
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = result["response"]

        # Confidence + category
        conf_class = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(r.confidence, "badge-medium")
        conf_icon  = {"HIGH": "✓", "MEDIUM": "~", "LOW": "✗"}.get(r.confidence, "~")
        cat_display = result["category"].replace("_", " ").title()

        st.markdown(
            f'<span class="{conf_class}">{conf_icon} {r.confidence}</span>'
            f'&nbsp;&nbsp;<span class="category-pill">{cat_display}</span>',
            unsafe_allow_html=True,
        )

        # Escalation banner
        if r.escalate:
            st.markdown(
                f'<div class="escalate-banner">⚠ Escalate to Team Leader — {r.escalation_reason}</div>',
                unsafe_allow_html=True,
            )

        # Customer need
        st.markdown('<div class="section-label">Customer need</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.9rem;color:#888;'>{r.customer_need}</div>", unsafe_allow_html=True)

        # Suggested response
        st.markdown('<div class="section-label">Suggested response</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="response-box">{r.suggested_response}</div>', unsafe_allow_html=True)

        # Key policy points
        if r.key_policy_points:
            st.markdown('<div class="section-label">Key policy points — agent reference only</div>', unsafe_allow_html=True)
            points_html = "".join(
                f'<div class="policy-point">• {p}</div>'
                for p in r.key_policy_points
            )
            st.markdown(f"<div style='background:#141E2B;border-radius:8px;padding:0.75rem 1rem;'>{points_html}</div>", unsafe_allow_html=True)

        # Sources
        if r.sources_used:
            st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
            st.markdown(
                " &nbsp; ".join(f'<code>{s}</code>' for s in r.sources_used),
                unsafe_allow_html=True,
            )

        # Retrieval details
        with st.expander("Show retrieval details", expanded=False):
            st.markdown("**Query rewritten as:**")
            st.markdown(f'<div class="rewrite-box">{result["rewritten_query"]}</div>', unsafe_allow_html=True)
            st.markdown("**Top retrieved chunks:**")
            for i, chunk in enumerate(result["retrieved_chunks"][:5], 1):
                sim   = chunk["similarity"]
                src   = chunk["source"]
                text  = chunk["text"][:300].replace("\n", " ")
                cls   = "" if sim >= 0.65 else "med" if sim >= 0.45 else "low"
                col   = "#1D9E75" if sim >= 0.65 else "#BA7517" if sim >= 0.45 else "#E24B4A"
                st.markdown(
                    f'<div class="chunk-card {cls}">'
                    f'<span style="font-size:0.75rem;color:{col};font-weight:600;">sim {sim}</span>'
                    f'&nbsp;<code>{src}</code><br>'
                    f'<span style="font-size:0.82rem;color:#888;">{text}...</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ Voltex Co-Pilot")
    st.markdown("""
    **Version:** 1.0.0  
    **Documents:** 5  
    **Chunks:** 421  
    **Model:** Claude Sonnet  
    **Embeddings:** all-MiniLM-L6-v2  
    """)
    st.divider()
    st.markdown("""
    **Confidence levels**
    - ✓ **HIGH** — clearly answered
    - ~ **MEDIUM** — use judgement
    - ✗ **LOW** — escalate or research
    """)
    st.divider()
    st.caption("Internal use only. Not for customer-facing use.")