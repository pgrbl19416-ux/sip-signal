
import streamlit as st
import json, re, os
from datetime import datetime

st.set_page_config(page_title="SIP Signal — AI Discovery Engine", layout="wide")

DATA_FILE = "sample_corpus.jsonl"
GOLDEN_FILE = "golden_set.json"

DEFAULT_PROMPT = """You are a product-discovery research analyst for Groww.
INTERVIEW_PROMPT = """
SYNTHESIS_PROMPT = """
You are a qualitative researcher synthesizing findings from interviews with
first-time SIP investors.

Using only the participant analyses below, identify patterns across participants.

Answer these research questions:
1. Why do people start a SIP, and what do they expect to happen?
2. What changes in their behavior in the months after starting?
3. What leads investors to pause, reduce, or stop a SIP?
4. What do investors check, how often, and how do they feel about it?
5. What confuses or worries them inside the app?
6. What do they do outside the app when unsure?
7. What separates investors who continue from those who quietly drop off?
8. How do patterns differ across income stability, age, source of money,
   and influence?
9. What unmet needs appear consistently?

Rules:
- Base every finding on the participant analyses.
- Mention how many participants support each finding, e.g. "4 of 6 participants".
- Include short supporting quotes.
- Clearly separate strong patterns from weak or single-participant observations.
- Do not invent participants, quotes, or patterns.

Return valid JSON:
{
  "research_questions": [
    {
      "question": "",
      "finding": "",
      "participant_count": 0,
      "supporting_quotes": [],
      "confidence": "strong | moderate | weak"
    }
  ],
  "segments": [
    {
      "segment": "",
      "pattern": "",
      "participant_ids": []
    }
  ],
  "unmet_needs": [
    {
      "need": "",
      "participant_count": 0,
      "evidence": []
    }
  ],
  "continuation_vs_drop_off": {
    "continuers": [],
    "drop_offs": [],
    "key_difference": ""
  },
  "product_opportunities": []
}

Participant analyses:
"""


INTERVIEW_PROMPT = """
You are a qualitative user-research analyst studying first-time SIP investors.

Analyze the interview notes below and extract evidence relevant to SIP continuation.

Identify:
1. Why the participant started a SIP and what they expected.
2. How their behavior changed after starting.
3. Whether they paused, reduced, stopped, or considered stopping, and why.
4. What they check in the app, how often, and how it makes them feel.
5. What confuses or worries them inside the app.
6. What they do outside the app when unsure.
7. Factors that may help them continue or cause them to drop off.
8. User-segment details: income stability, age, source of money, and influences.
9. Unmet needs or recurring pain points.

Rules:
- Use only information explicitly present in the notes.
- Quote the participant exactly where possible.
- If something is not mentioned, write "Not mentioned".
- Do not invent insights.
- Separate facts from interpretation.

Return valid JSON in this format:
{
  "participant_id": "",
  "segment": {
    "age": "",
    "occupation": "",
    "income_stability": "",
    "source_of_money": "",
    "influenced_by": ""
  },
  "why_started": {
    "motivation": "",
    "expectations": "",
    "evidence": ""
  },
  "behavior_after_start": {
    "checking_frequency": "",
    "what_they_check": "",
    "emotional_response": "",
    "evidence": ""
  },
  "pause_reduce_stop": {
    "status": "",
    "trigger": "",
    "decision_process": "",
    "evidence": ""
  },
  "in_app_experience": {
    "confusions": [],
    "worries": [],
    "evidence": []
  },
  "outside_app_behavior": {
    "sources": [],
    "people_consulted": [],
    "evidence": []
  },
  "continuation_drivers": [],
  "drop_off_risks": [],
  "unmet_needs": [],
  "notable_quotes": []
}

Interview notes:
"""
Analyse the public post below about SIPs / mutual-fund investing.

Return ONLY valid JSON with these fields:
{
  "relevant": true/false,
  "segment": "first-time investor | salaried | freelancer | student | other",
  "sip_stage": "before start | months 1-3 | months 4-6 | months 7-12 | after 12 months | unclear",
  "expectation": "short phrase",
  "trigger": "market fall | return disappointment | cash-flow stress | payment failure | confusion | social influence | emergency | none",
  "behaviour": "started | continued | reduced | paused | stopped | redeemed | considering pause | none",
  "in_app_action": "checked portfolio | contacted support | changed SIP | paused | cancelled | none",
  "outside_app_action": "asked friends/family | searched YouTube | searched Reddit | read news | none",
  "emotion": "calm | confused | anxious | disappointed | neutral",
  "continuation_risk": "low | medium | high",
  "theme": "short theme name",
  "evidence_quote": "short verbatim quote",
  "confidence": 0.0-1.0
}

Rules:
- Do not invent facts not present in the post.
- If the post is irrelevant, set relevant=false and use "none"/"unclear" where appropriate.
- Do not provide investment advice.

POST:
"""

@st.cache_data
def load_corpus():
    rows=[]
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip(): rows.append(json.loads(line))
    return rows

@st.cache_data
def load_golden():
    if os.path.exists(GOLDEN_FILE):
        with open(GOLDEN_FILE, encoding="utf-8") as f: return json.load(f)
    return []

def rule_based_extract(text):
    t = text.lower()

    relevant = any(k in t for k in [
        "sip", "mutual fund", "invest", "groww",
        "instalment", "installment"
    ])

    trigger = "none"
    trigger_signals = 0

    trigger_rules = [
        ("market fall", "market fall"),
        ("loss", "return disappointment"),
        ("negative", "return disappointment"),
        ("salary", "cash-flow stress"),
        ("emergency", "emergency"),
        ("failed", "payment failure"),
        ("confus", "confusion"),
        ("reddit", "social influence"),
        ("youtube", "social influence")
    ]

    for keyword, label in trigger_rules:
        if keyword in t:
            trigger = label
            trigger_signals += 1
            break

    behaviour = "none"
    behaviour_rules = [
        ("pause", "paused"),
        ("stop", "stopped"),
        ("cancel", "stopped"),
        ("redeem", "redeemed"),
        ("continue", "continued"),
        ("start", "started")
    ]

    behaviour_signals = 0

    for keyword, label in behaviour_rules:
        if keyword in t:
            behaviour = label
            behaviour_signals += 1
            break

    risk = "low"

    if behaviour in ["paused", "stopped", "redeemed"]:
        risk = "high"
    elif trigger in [
        "cash-flow stress",
        "payment failure",
        "return disappointment"
    ]:
        risk = "high"
    elif trigger != "none":
        risk = "medium"

    # Calculate a simple confidence score
    base_confidence = 0.40

    if relevant:
        base_confidence += 0.20

    if trigger != "none":
        base_confidence += 0.15

    if behaviour != "none":
        base_confidence += 0.15

    if risk == "high":
        base_confidence += 0.05

    confidence = round(min(base_confidence, 0.90), 2)

    return {
        "relevant": relevant,
        "segment": "first-time investor" if "first" in t else "other",
        "sip_stage": "unclear",
        "expectation": "unclear",
        "trigger": trigger,
        "behaviour": behaviour,
        "in_app_action": "none",
        "outside_app_action": "none",
        "emotion": "anxious" if any(
            word in t for word in ["worri", "anx", "panic", "stress"]
        ) else "neutral",
        "continuation_risk": risk,
        "theme": trigger if trigger != "none" else "general SIP experience",
        "evidence_quote": text[:180],
        "confidence": confidence
    }
def gemini_extract(text, api_key, model="gemini-3.8-flash"):
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    wait_times = [2, 5, 10]

    for attempt, wait_time in enumerate(wait_times, start=1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=DEFAULT_PROMPT + text,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )

            return json.loads(response.text)

        except Exception as e:
            error_text = str(e)

            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < len(wait_times):
                    st.warning(
                        f"Gemini is busy. Retrying in {wait_time} seconds "
                        f"(attempt {attempt + 1} of {len(wait_times)})."
                    )
                    time.sleep(wait_time)
                    continue

            st.error(f"Gemini extraction failed: {e}")
            return None

    return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=DEFAULT_PROMPT + text,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Gemini extraction failed: {e}")
        return None
def llm_extract(text, provider, api_key, model):
    try:
        if provider=="OpenAI":
            from openai import OpenAI
            client=OpenAI(api_key=api_key)
            resp=client.chat.completions.create(model=model, temperature=0,
                response_format={"type":"json_object"},
                messages=[{"role":"user","content":DEFAULT_PROMPT+text}])
            return json.loads(resp.choices[0].message.content)
        if provider=="Anthropic":
            import anthropic
            client=anthropic.Anthropic(api_key=api_key)
            resp=client.messages.create(model=model, max_tokens=800, temperature=0,
                system="Return only valid JSON.",
                messages=[{"role":"user","content":DEFAULT_PROMPT+text}])
            return json.loads(resp.content[0].text)
    except Exception as e:
        st.error(f"LLM extraction failed: {e}")
    return None

st.title("SIP Signal")
st.caption("AI-powered discovery engine for first-year SIP continuation · Educational research prototype · Not investment advice")

with st.sidebar:
    st.header("Engine settings")
    provider=st.selectbox("LLM provider", ["Rule-based demo","OpenAI","Anthropic","Gemini"])
    api_key=st.text_input("API key", type="password")
if provider == "OpenAI":
    default_model = "gpt-4o-mini"
elif provider == "Anthropic":
    default_model = "claude-3-5-haiku-latest"
elif provider == "Gemini":
    default_model = "gemini-3.8-flash"
else:
    default_model = ""

model=st.text_input("Model", value=default_model)
st.divider()
st.write("Use the rule-based demo without an API key. Add a key to run live LLM extraction.")

rows=load_corpus()
st.success(f"Loaded {len(rows)} public items from the sample corpus.")

col1,col2=st.columns([2,1])
with col1:
    query=st.text_input("Search corpus", placeholder="e.g. pause, negative return, salary delay")
with col2:
    risk_filter=st.multiselect("Continuation risk", ["low","medium","high"], default=[])

filtered = []

for r in rows:
    # Search filter
    if query and query.lower() not in json.dumps(r).lower():
        continue

    # Find continuation risk safely
    item_risk = r.get("continuation_risk")

    if item_risk is None:
        extracted = r.get("extracted") or {}
        item_risk = extracted.get("continuation_risk")

    # Apply risk filter only when user has selected something
    if risk_filter:
        if item_risk is None:
            continue
        if item_risk not in risk_filter:
            continue

    filtered.append(r)
st.subheader(f"Corpus ({len(filtered)} shown)")
for r in filtered:
    with st.expander(f"{r.get('platform','unknown')} · {r.get('date','')} · {r.get('theme') or (r.get('extracted') or {}).get('theme') or 'unprocessed'}"):
        st.write(r.get("text",""))
        st.json(r.get("extracted", {}), expanded=False)
        if r.get("source_url"): st.markdown(f"[Source]({r['source_url']})")

st.divider()
st.subheader("Run extraction")
idx=st.number_input("Corpus row", min_value=0, max_value=max(0,len(rows)-1), value=0)
if st.button("Extract insights"):
    text=rows[idx]["text"]

    if provider=="Rule-based demo":
        result=rule_based_extract(text)

    elif provider=="Gemini":
        result=gemini_extract(text, api_key, model) or rule_based_extract(text)

    else:
        result=llm_extract(text, provider, api_key, model) or rule_based_extract(text)
    st.json(result)
    st.download_button("Download extraction JSON", json.dumps(result, indent=2), file_name="sip_signal_extraction.json")

st.divider()
st.subheader("Golden-set evaluation")
golden=load_golden()
if golden:
    agree=sum(1 for g in golden if g.get("engine_agrees"))
    st.metric("Engine agreement with hand labels", f"{agree}/{len(golden)}", f"{round(100*agree/len(golden))}%")
    st.dataframe(golden)
else:
    st.info("Add golden_set.json with 20 hand-labelled examples.")
