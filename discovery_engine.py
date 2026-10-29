import json
import re
import pandas as pd
import streamlit as st
from google import genai

st.set_page_config(
    page_title="SIP Discovery Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SIP Discovery Engine")
st.caption(
    "Explore evidence from real investor discussions. "
    "Educational research only — not investment advice."
)

# -------------------------
# Gemini setup
# -------------------------
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    client = None
    st.warning("Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit Cloud secrets.")

MODEL_NAME = "gemini-2.5-flash"

# -------------------------
# Evidence upload
# -------------------------
st.subheader("1. Load Evidence Log")

uploaded_file = st.file_uploader(
    "Upload your Evidence Log CSV",
    type=["csv"],
    help="Required columns: post_id, source_title, source_url, date, post_text"
)

if uploaded_file is not None:
    try:
        evidence_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
else:
    st.info("Upload an Evidence Log CSV to begin.")
    st.stop()

required_columns = ["post_id", "source_title", "source_url", "date", "post_text"]
missing = [c for c in required_columns if c not in evidence_df.columns]

if missing:
    st.error(f"Your CSV is missing required columns: {', '.join(missing)}")
    st.stop()

evidence_df = evidence_df.fillna("")
st.success(f"Loaded {len(evidence_df)} evidence records.")

# -------------------------
# User selections
# -------------------------
st.subheader("2. Choose what you want to explore")

situation = st.selectbox(
    "What happened?",
    [
        "My SIP payment or auto-debit failed",
        "My salary was delayed",
        "I am worried because my SIP is showing a loss",
        "I am thinking about stopping my SIP",
        "Something else"
    ]
)

concern_map = {
    "My SIP payment or auto-debit failed": [
        "Common investor experiences",
        "Possible reasons for the failure",
        "Questions to ask official support",
        "What to avoid deciding in panic"
    ],
    "My salary was delayed": [
        "Common investor experiences",
        "Cash-flow pressure and SIP continuity",
        "Questions to ask official support",
        "What to avoid deciding in panic"
    ],
    "I am worried because my SIP is showing a loss": [
        "Common investor experiences",
        "Market decline and long-term investing",
        "Questions to ask official support",
        "What to avoid deciding in panic"
    ],
    "I am thinking about stopping my SIP": [
        "Common investor experiences",
        "Reasons investors consider stopping",
        "Questions to ask official support",
        "What to avoid deciding in panic"
    ],
    "Something else": [
        "General SIP experiences"
    ]
}

concern = st.selectbox(
    "What do you want to understand?",
    concern_map[situation]
)

# -------------------------
# Simple evidence retrieval
# -------------------------
def retrieve_evidence(df, situation, concern, limit=8):
    situation_keywords = {
        "My SIP payment or auto-debit failed": [
            "auto debit", "autodebit", "payment failed", "payment failure",
            "mandate", "insufficient balance", "sip failed", "bounced"
        ],
        "My salary was delayed": [
            "salary", "salary delayed", "cash flow", "cashflow",
            "financial emergency", "job loss", "income"
        ],
        "I am worried because my SIP is showing a loss": [
            "loss", "losses", "market fall", "market crash", "negative returns",
            "red", "down", "volatility", "panic"
        ],
        "I am thinking about stopping my SIP": [
            "stop sip", "stopped sip", "pause sip", "discontinue",
            "redeem", "exit", "continue sip"
        ],
        "Something else": []
    }

    concern_keywords = {
        "Common investor experiences": ["experience", "happened", "faced", "issue"],
        "Possible reasons for the failure": ["reason", "because", "failed", "issue"],
        "Cash-flow pressure and SIP continuity": ["cash flow", "salary", "emergency", "continue"],
        "Market decline and long-term investing": ["long term", "long-term", "market", "volatility"],
        "Reasons investors consider stopping": ["stop", "pause", "exit", "redeem"],
        "Questions to ask official support": ["support", "customer care", "ask", "confirm"],
        "What to avoid deciding in panic": ["panic", "emotional", "decide", "avoid"],
        "General SIP experiences": []
    }

    keywords = situation_keywords.get(situation, []) + concern_keywords.get(concern, [])
    keywords = [k.lower() for k in keywords]

    if not keywords:
        return df.head(limit).copy()

    text_columns = ["post_text", "source_title"]
    scores = []

    for _, row in df.iterrows():
        combined = " ".join(str(row[c]) for c in text_columns).lower()
        score = sum(combined.count(k) for k in keywords)
        scores.append(score)

    df = df.copy()
    df["relevance_score"] = scores
    filtered = df[df["relevance_score"] > 0].sort_values(
        "relevance_score", ascending=False
    )

    if filtered.empty:
        return df.head(limit).copy()

    return filtered.head(limit).copy()

# -------------------------
# Generate insight
# -------------------------
st.subheader("3. Generate insight")

if st.button("Find insights", type="primary", use_container_width=True):
    if client is None:
        st.error("Gemini API key is missing. Add it in Streamlit Cloud secrets.")
        st.stop()

    if situation == "Something else":
        st.warning(
            "This request is outside the supported SIP evidence topics. "
            "Please choose one of the available situations."
        )
        st.stop()

    with st.spinner("Retrieving evidence and generating insight..."):
        relevant = retrieve_evidence(evidence_df, situation, concern)

        evidence_for_model = relevant[
            ["post_id", "source_title", "source_url", "date", "post_text"]
        ].to_dict(orient="records")

        prompt = f"""
You are an evidence-based SIP research assistant.

Use ONLY the evidence provided below.

User situation: {situation}
User concern: {concern}

Return valid JSON only with these keys:
{{
  "summary": "2-4 sentence evidence-based explanation",
  "themes": ["short theme labels"],
  "quotes": [
    {{
      "quote": "exact short quote from a source",
      "post_id": "matching post_id",
      "source_title": "matching source_title",
      "source_url": "matching source_url"
    }}
  ],
  "questions": ["questions the user can ask official support"],
  "uncertainty": "clear statement of uncertainty",
  "non_advice_disclaimer": "This is educational research, not investment advice."
}}

Rules:
- Use only information present in the evidence.
- Copy quotes exactly from the evidence.
- Do not invent quotes, sources, dates, or facts.
- Use at most 3 quotes.
- Do not recommend buying, selling, holding, pausing, continuing, redeeming, or exiting.
- If evidence is insufficient, clearly say so in summary and uncertainty.

Evidence:
{json.dumps(evidence_for_model, ensure_ascii=False)}
"""

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )

            raw = response.text.strip()
            raw = re.sub(r"^```json|```$", "", raw).strip()
            insight = json.loads(raw)

            st.session_state["current_insight"] = insight
            st.session_state["current_evidence"] = evidence_for_model

        except Exception as e:
            st.error("The insight could not be generated. Please try again.")
            st.code(str(e))
            st.stop()

# -------------------------
# Display insight
# -------------------------
if "current_insight" in st.session_state:
    insight = st.session_state["current_insight"]

    st.subheader("Insight")

    st.markdown("### Main pattern")
    st.write(insight.get("summary", "No summary available."))

    themes = insight.get("themes", [])
    if themes:
        st.markdown("### Themes")
        st.write(", ".join(themes))

    quotes = insight.get("quotes", [])
    if quotes:
        st.markdown("### Supporting evidence")

        for q in quotes:
            with st.container(border=True):
                st.markdown(f"> {q.get('quote', '')}")
                st.caption(
                    f"Source: {q.get('source_title', 'Unknown source')} — "
                    f"{q.get('source_url', 'No link')}"
                )
    else:
        st.info("No direct quotes were available in the retrieved evidence.")

    questions = insight.get("questions", [])
    if questions:
        st.markdown("### Questions for official support")
        for question in questions:
            st.write(f"- {question}")

    st.markdown("### Uncertainty")
    st.write(insight.get("uncertainty", "Evidence may be incomplete."))

    st.info(insight.get(
        "non_advice_disclaimer",
        "Educational research only. This is not investment advice."
    ))

    # -------------------------
    # Save to golden set
    # -------------------------
    st.subheader("4. Save for golden-set review")

    if st.button("Save this insight for review", use_container_width=True):
        case_id = f"discovery_case_{len(st.session_state.get('golden_cases', [])) + 1:03d}"

        golden_case = {
            "id": case_id,
            "situation": situation,
            "user_details": concern,
            "expected_themes": insight.get("themes", []),
            "expected_behavior": (
                "Provide evidence-based educational explanation, express uncertainty, "
                "use grounded quotes, and avoid investment advice."
            ),
            "expected_sources": [
                q.get("source_url", "")
                for q in insight.get("quotes", [])
                if q.get("source_url")
            ],
            "quotes": insight.get("quotes", []),
            "questions": insight.get("questions", []),
            "review_status": "needs_review",
            "created_by": "discovery_engine"
        }

        if "golden_cases" not in st.session_state:
            st.session_state["golden_cases"] = []

        st.session_state["golden_cases"].append(golden_case)
        st.success(f"Saved as {case_id}. Review before exporting.")

# -------------------------
# Golden-set review and export
# -------------------------
st.subheader("5. Golden-set review")

golden_cases = st.session_state.get("golden_cases", [])

if not golden_cases:
    st.info("No insights saved yet.")
else:
    for i, case in enumerate(golden_cases):
        with st.expander(f"{case['id']} — {case['situation']}"):
            st.write(f"**User concern:** {case['user_details']}")
            st.write(f"**Expected themes:** {', '.join(case['expected_themes'])}")
            st.write(f"**Review status:** {case['review_status']}")

            approve = st.checkbox(
                "Approve this case for golden set",
                key=f"approve_{i}"
            )

            if approve:
                golden_cases[i]["review_status"] = "approved"
            else:
                golden_cases[i]["review_status"] = "needs_review"

    approved_cases = [c for c in golden_cases if c["review_status"] == "approved"]

    st.write(f"**Approved cases:** {len(approved_cases)} of {len(golden_cases)}")

    if approved_cases:
        jsonl_data = "\n".join(
            json.dumps(case, ensure_ascii=False) for case in approved_cases
        )

        st.download_button(
            label="Download approved golden_set.jsonl",
            data=jsonl_data,
            file_name="golden_set.jsonl",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.warning("Approve at least one case before exporting.")

st.divider()
st.caption(
    "SIP Discovery Engine is an educational research tool. "
    "It does not provide investment advice."
)
