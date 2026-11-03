import json
import os
import re
import time
import hashlib
import inspect

import streamlit as st
from google import genai
from google.genai import types

DATA_FILE = "auto_tagged_corpus.jsonl"
GOLDEN_FILE = "golden_set.jsonl"

MODEL_NAME = "gemini-3.5-flash-lite"
GOLDEN_PASS_THRESHOLD = 0.80
GATE_VERSION = "sip-assistant-gate-v1"

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

    if not isinstance(result, dict):
        return ["AI output must be a JSON object."]

    if result.get("relevant") is not True:
        issues.append("Response was not marked relevant.")

    required_list_fields = [
        "identified_themes",
        "common_patterns",
        "questions_to_consider",
        "safe_next_steps",
    ]

    for field in required_list_fields:
        value = result.get(field)

        if (
            not isinstance(value, list)
            or not value
            or not all(
                isinstance(item, str) and item.strip()
                for item in value
            )
        ):
            issues.append(
                f"{field} must be a nonempty list of strings."
            )

    quotes = result.get("evidence_quotes", [])

    def normalize_text(value):
        return " ".join(value.split()).casefold()

    if not isinstance(quotes, list) or not quotes:
        issues.append("No valid evidence quotes provided.")

    else:
        for index, quote in enumerate(quotes, start=1):
            if not isinstance(quote, dict):
                issues.append(
                    f"Evidence quote {index} is not an object."
                )
                continue

            quote_text = quote.get("quote")
            source_url = quote.get("source_url")

            if (
                not isinstance(quote_text, str)
                or not quote_text.strip()
                or not isinstance(source_url, str)
                or not source_url.strip()
            ):
                issues.append(
                    f"Evidence quote {index} needs text "
                    "and a source URL."
                )
                continue

            matching_sources = [
                row
                for row in evidence
                if row.get("source_url") == source_url
            ]

            verified = any(
                normalize_text(quote_text)
                in normalize_text(
                    row.get("text", "")[:1200]
                )
                for row in matching_sources
            )

            if not verified:
                issues.append(
                    f"Evidence quote {index} was not found "
                    "in the provided text for its claimed source."
                )

    for field in ("uncertainty", "disclaimer"):
        value = result.get(field)

        if not isinstance(value, str) or not value.strip():
            issues.append(
                f"{field} must be a nonempty string."
            )

    forbidden_patterns = [
        r"\byou should (?:invest|redeem|sell|buy|stop|pause|continue)\b",
        r"\bi recommend (?:investing|redeeming|selling|buying|stopping|pausing|continuing)\b",
        r"\byou must (?:invest|redeem|sell|buy|stop|pause|continue)\b",
    ]

    # Inspect generated guidance, not quoted public-post text.
    guidance_fields = [
        "identified_themes",
        "common_patterns",
        "questions_to_consider",
        "uncertainty",
        "safe_next_steps",
        "disclaimer",
    ]

    answer_text = json.dumps(
        {
            field: result.get(field)
            for field in guidance_fields
        },
        ensure_ascii=False
    ).lower()

    if any(
        re.search(pattern, answer_text)
        for pattern in forbidden_patterns
    ):
        issues.append(
            "Response appears to contain direct investment advice."
        )

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
        return {
            "passed": False,
            "pass_rate": 0.0,
            "total": 0,
            "results": [],
        }

    passed = 0
    results = []

    progress = st.progress(
        0,
        text="Running golden-set evaluation..."
    )

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
                        parts=[
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )

            result = parse_ai_response(response.text)
            issues = quality_check(result, evidence)

            expected_themes = set(
                case.get("expected_themes", [])
            )
            identified_themes = set(
                result.get("identified_themes", [])
            )

            theme_overlap = len(
                expected_themes.intersection(
                    identified_themes
                )
            )

            minimum_quotes = case.get(
                "minimum_evidence_quotes",
                1
            )

            if theme_overlap == 0:
                issues.append(
                    "No identified theme matches the expected themes."
                )

            if (
                len(result.get("evidence_quotes", []))
                < minimum_quotes
            ):
                issues.append(
                    "Fewer evidence quotes than required: "
                    f"minimum {minimum_quotes}."
                )

            case_passed = not issues

            if case_passed:
                passed += 1

            results.append({
                "id": case["id"],
                "passed": case_passed,
                "issues": issues,
                "identified_themes": sorted(identified_themes),
                "expected_themes": sorted(expected_themes),
            })

        except Exception as error:
            results.append({
                "id": case["id"],
                "passed": False,
                "issues": [
                    f"API, parsing, or validation error: {error}"
                ],
                "identified_themes": [],
                "expected_themes": case.get(
                    "expected_themes",
                    []
                ),
            })

        progress.progress(
            index / len(golden_cases),
            text=(
                f"Evaluated {index} "
                f"of {len(golden_cases)} cases"
            )
        )

        time.sleep(1)

    accuracy = passed / len(golden_cases)

    st.metric(
        "Golden-set pass rate",
        f"{accuracy:.0%}"
    )

    for result in results:
        status = (
            "✅ Pass"
            if result["passed"]
            else "❌ Fail"
        )

        st.write(
            f"{status} — {result['id']} | "
            f"Expected: {', '.join(result['expected_themes'])} | "
            f"Identified: {', '.join(result['identified_themes'])}"
        )

        if result["issues"]:
            st.write(
                f"Issues: {'; '.join(result['issues'])}"
            )

    # This block is outside both loops.
    gate_passed = accuracy >= GOLDEN_PASS_THRESHOLD

    if gate_passed:
        st.success(
            "Golden-set quality gate passed. "
            "The assistant is enabled for this configuration."
        )
    else:
        st.error(
            "Golden-set quality gate failed. "
            "New AI reflections are blocked."
        )

    return {
        "passed": gate_passed,
        "pass_rate": accuracy,
        "total": len(golden_cases),
        "results": results,
    }


def current_gate_fingerprint():
    payload = {
        "version": GATE_VERSION,
        "model": MODEL_NAME,
        "threshold": GOLDEN_PASS_THRESHOLD,
        "temperature": 0.2,
        "code": {
            function.__name__: inspect.getsource(function)
            for function in (
                retrieve_evidence,
                build_prompt,
                parse_ai_response,
                quality_check,
                run_golden_evaluation,
            )
        },
        "files": {},
    }

    for path in (DATA_FILE, GOLDEN_FILE):
        if os.path.exists(path):
            with open(path, "rb") as file:
                payload["files"][path] = hashlib.sha256(
                    file.read()
                ).hexdigest()
        else:
            payload["files"][path] = None

    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()

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

        fingerprint = current_gate_fingerprint()

    if evaluate:
        # Remove any previous approval before starting a fresh evaluation.
        st.session_state.pop("assistant_quality_gate", None)

        # Refresh cached files before evaluating their current contents.
        load_corpus.clear()
        load_golden_set.clear()
        corpus = load_corpus()

        try:
            report = run_golden_evaluation()
        except Exception as error:
            st.error(f"Evaluation could not complete: {error}")
            report = {
                "passed": False,
                "pass_rate": 0.0,
                "total": 0,
                "results": [],
            }

        report["fingerprint"] = fingerprint
        st.session_state["assistant_quality_gate"] = report

    gate = st.session_state.get("assistant_quality_gate")

    gate_is_current = (
        gate is not None
        and gate.get("fingerprint") == fingerprint
    )

    gate_passed = (
        gate_is_current
        and gate.get("passed") is True
    )

    if gate_passed:
        st.success(
            "Evaluation gate: passed "
            f"({gate['pass_rate']:.0%} case pass rate). "
            "Each new response will also be checked before display."
        )
    elif gate_is_current:
        st.warning(
            "Evaluation gate: failed. "
            "New AI reflections are blocked."
        )
    else:
        st.info(
            "Evaluation gate: not evaluated for the current configuration. "
            "Run the golden-set evaluation before generating a reflection."
        )

    if generate and not gate_passed:
        safe_fallback(
            "The current assistant configuration has not passed "
            "the golden-set evaluation."
        )
        return

    if not generate:
        if gate_passed:
            st.info(
                "Choose a situation and click "
                "Generate evidence-based reflection."
            )
        elif gate_is_current:
            st.info(
                "Review the failed evaluation cases. "
                "Resolve the issues and rerun evaluation "
                "before generating a reflection."
            )
        else:
            st.info(
                "Run the golden-set evaluation first. "
                "Generation is available only after "
                "the current configuration passes."
            )

        st.warning(
            "This tool does not provide investment advice, "
            "recommendations, or predictions."
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
