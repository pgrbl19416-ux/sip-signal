# SIP Signal — AI-Powered Discovery Engine (Prototype)

## What it does
SIP Signal ingests public conversations about SIPs and first-time investing, extracts structured
behavioural signals, clusters recurring themes, and ranks continuation-risk opportunities for Groww.

## Run locally
```bash
cd output/sip_signal
pip install streamlit
streamlit run app.py
```

## Modes
- Rule-based demo: works without API keys, useful for demonstrating workflow.
- OpenAI / Anthropic: paste an API key in the sidebar to run live LLM extraction.

## Data
`sample_corpus.jsonl` contains illustrative public-style posts. Replace/augment it with collected
Google Play, App Store, Reddit, TradingQ&A, YouTube, and X data through your collection workflow.

## Evidence log
`golden_set.json` is the start of the hand-labelled golden set. Expand it to 20 items, record:
- hand label
- engine output
- agreement
- failure reason
- correction made

## Safety
This tool is for product research. It does not provide investment advice, recommend funds,
predict returns, or use fear/FOMO/urgency nudges.
