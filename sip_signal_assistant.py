import json
import os
import re
import time

import streamlit as st
from google import genai
from google.genai import types

DATA_FILE = "auto_tagged_corpus.jsonl"
GOLDEN_FILE = "golden_set.jsonl"

MODEL_NAME = "gemini-3.5-flash-lite"

st.set_page_config(
    page_title="SIP Signal Assistant",
    page_icon="📊",
    layout="wide"
)

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_data
def load_corpus():
    records = []

    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

    return records


@st.cache_data
def load_golden_set():
    records = []

    if os.path.exists(GOLDEN_FILE):
        with open(GOLDEN_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

    return records


def retrieve_evidence(corpus, situation, selected_tags, limit=6):
    situation_words = set(
        word for word in re.findall(r"[a-zA-Z]{3,}", situation.lower())
    )

    scored = []

    for row in corpus:
        tags = row.get("auto_tags", [])
        text = row.get("text", "").lower()

        score = 0

        for tag in selected_tags:
            if tag in tags:
                score += 3

        for word in situation_words:
            if word in text:
                score += 1

        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [row for score, row in scored[:limit]]


def build_prompt(situation, selected_tags, evidence):
    evidence_text = ""

    for index, row in enumerate(evidence, start=1):
        evidence_text += f"""
Evidence {index}
Source: {row.get("source_url", "Unknown")}
Platform: {row.get("platform", "Unknown")}
Tags: {", ".join(row.get("auto_tags", []))}
Post: {row.get("text", "")[:1200]}
"""

    return f"""
You are the SIP Signal Assistant.

Help a first-time SIP investor reflect on public evidence about SIP continuation,
pausing, stopping, returns, market shocks, payment issues, and app experience.

STRICT RULES:
- Do NOT give personalized investment advice.
- Do NOT tell the user to invest, redeem, sell, buy, stop, pause, or continue.
- Use only the provided evidence.
- If evidence is insufficient, say so clearly.
- Do not invent quotes or facts.
- Mention that official fund documents and a SEBI-registered investment adviser
should be consulted for personalized decisions.

Return ONLY valid JSON:

{{
  "relevant": true,
  "identified_themes": [],
  "common_patterns": [],
  "questions_to_consider": [],
  "evidence_quotes": [
    {{"quote": "", "source_url": "", "platform": ""}}
  ],
  "uncertainty": "",
  "safe_next_steps": [],
  "disclaimer": "This is educational information, not investment advice."
}}

User situation:
{situation}

Selected evidence themes:
{", ".join(selected_tags)}

Evidence:
{evidence_text}
"""


def parse_ai_response(response_text):
    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned)

    return json.loads(cleaned)


def quality_check(result, evidence):
    issues = []

    if not result.get("relevant", False):
        issues.append("Response marked irrelevant.")

    if not result.get("identified_themes"):
        issues.append("No themes identified.")

    if not result.get("common_patterns"):
        issues.append("No patterns provided.")

    if not result.get("questions_to_consider"):
        issues.append("No reflection questions provided.")

    quotes = result.get("evidence_quotes", [])

    if not quotes:
        issues.append("No evidence quotes provided.")

    if evidence and not quotes:
        issues.append("Evidence exists but was not cited.")

    forbidden_patterns = [
        r"\byou should (?:invest|redeem|sell|buy|stop|pause|continue)\b",
        r"\bi recommend (?:investing|redeeming|selling|buying|stopping|pausing|continuing)\b",
        r"\byou must (?:invest|redeem|sell|buy|stop|pause|continue)\b"
    ]

    answer_text = json.dumps(result).lower()

    if any(
        re.search(pattern, answer_text)
        for pattern in forbidden_patterns
    ):
        issues.append("Response appears to contain direct investment advice.")

    return issues


def safe_fallback(reason):
    st.warning("## Safe fallback")

    st.write(
        "I could not generate a sufficiently reliable evidence-based reflection "
        "for this situation."
    )

    st.write(f"**Quality check:** {reason}")

    st.info(
        "For decisions about your own investments, refer to official fund documents, "
        "your platform’s support resources, and a SEBI-registered investment adviser."
    )


def run_golden_evaluation():
    corpus = load_corpus()
    golden_cases = load_golden_set()

    if not golden_cases:
        st.warning("No golden-set cases found.")
        return

    passed = 0
    results = []

    progress = st.progress(0, text="Running golden-set evaluation...")

    for index, case in enumerate(golden_cases, start=1):
        evidence = retrieve_evidence(
            corpus,
            case["user_situation"],
            case.get("expected_themes", [])
        )

        prompt = build_prompt(
            case["user_situation"],
            case.get("expected_themes", []),
            evidence
        )

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )

            result = parse_ai_response(response.text)

            issues = quality_check(result, evidence)

            expected_themes = set(case.get("expected_themes", []))
            identified_themes = set(result.get("identified_themes", []))

            theme_overlap = len(
                expected_themes.intersection(identified_themes)
            )

            case_passed = (
                not issues
                and theme_overlap > 0
                and len(result.get("evidence_quotes", []))
                >= case.get("minimum_evidence_quotes", 1)
            )

            if case_passed:
                passed += 1

            results.append({
                "id": case["id"],
                "passed": case_passed,
                "issues": issues,
                "identified_themes": list(identified_themes),
                "expected_themes": list(expected_themes)
            })

        except Exception as error:
            results.append({
                "id": case["id"],
                "passed": False,
                "issues": [f"API or parsing error: {error}"],
                "identified_themes": [],
                "expected_themes": case.get("expected_themes", [])
            })

        progress.progress(
            index / len(golden_cases),
            text=f"Evaluated {index} of {len(golden_cases)} cases"
        )

        time.sleep(1)

    accuracy = passed / len(golden_cases)

    st.metric("Golden-set pass rate", f"{accuracy:.0%}")

    for result in results:
        status = "✅ Pass" if result["passed"] else "❌ Fail"

        st.write(
            f"{status} — {result['id']} | "
            f"Expected: {', '.join(result['expected_themes'])} | "
            f"Identified: {', '.join(result['identified_themes'])}"
        )

        if result["issues"]:
            st.write(f"Issues: {', '.join(result['issues'])}")


def main():
    st.title("📊 SIP Signal Assistant")

    st.caption(
        "Evidence-based reflection support for first-time SIP investors. "
        "This is educational information, not investment advice."
    )

    corpus = load_corpus()

    if not corpus:
        st.error(
            "No corpus file found. Please upload `auto_tagged_corpus.jsonl` "
            "to the repository."
        )
        return

    with st.sidebar:
        st.header("Your situation")

        situation_options = [
            "My SIP payment or auto-debit failed",
            "My salary is delayed or I have a cash-flow problem",
            "The market has fallen and I am anxious",
            "I am confused about whether to pause, stop, or continue",
            "I have an app or onboarding issue",
            "Other"
        ]

        situation = st.selectbox(
            "What best describes your situation?",
            situation_options
        )

        details = st.text_area(
            "Add optional details",
            placeholder="Example: My SIP auto-debit failed this month and I am unsure what happens next."
        )

        tag_options = sorted(
            set(
                tag
                for row in corpus
                for tag in row.get("auto_tags", [])
            )
        )

        selected_tags = st.multiselect(
            "Evidence themes to prioritize",
            tag_options,
            default=[]
        )

        generate = st.button("Generate evidence-based reflection")

        st.divider()

        st.subheader("Quality evaluation")

        st.caption(
            "Tests the AI against your golden set and shows pass/fail results."
        )

        evaluate = st.button("Run golden-set evaluation")

    if evaluate:
        run_golden_evaluation()

    if not generate:
        st.info(
            "Choose a situation and click **Generate evidence-based reflection**."
        )

        st.warning(
            "This tool does not provide investment advice, recommendations, "
            "or predictions."
        )

        return

    full_situation = f"{situation}. {details}".strip()

    evidence = retrieve_evidence(
        corpus,
        full_situation,
        selected_tags
    )

    if not evidence:
        safe_fallback("No relevant evidence was found in the corpus.")
        return

    prompt = build_prompt(
        full_situation,
        selected_tags,
        evidence
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        result = parse_ai_response(response.text)

    except Exception as error:
        safe_fallback(f"Gemini request failed: {error}")
        return

    issues = quality_check(result, evidence)

    if issues:
        safe_fallback("; ".join(issues))
        return

    st.success("Quality checks passed.")

    st.subheader("Patterns seen in public investor discussions")

    for pattern in result.get("common_patterns", []):
        st.write(f"- {pattern}")

    st.subheader("Questions to consider")

    for question in result.get("questions_to_consider", []):
        st.write(f"- {question}")

    st.subheader("Evidence from public posts")

    for quote in result.get("evidence_quotes", []):
        st.markdown(
            f"> “{quote.get('quote', '')}”  \n"
            f"— [{quote.get('platform', 'Source')}]({quote.get('source_url', '#')})"
        )

    st.subheader("Uncertainty")

    st.info(
        result.get(
            "uncertainty",
            "Public posts may not represent all investor experiences."
        )
    )

    st.subheader("General, non-personalized considerations")

    for step in result.get("safe_next_steps", []):
        st.write(f"- {step}")

    st.error(
        result.get(
            "disclaimer",
            "This is educational information, not investment advice."
        )
    )

    st.subheader("Official resources")

    st.write(
        "- [SEBI Investor](https://investor.sebi.gov.in/)  \n"
        "- [AMFI](https://www.amfiindia.com/)"
    )


if __name__ == "__main__":
    main()
