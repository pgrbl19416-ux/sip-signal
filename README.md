# SIP Signal — improved Discovery Engine

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Workspaces
- Overview: workflow, evidence boundaries and data limitations.
- Public-post analysis: structured extraction with quote/schema validation.
- Interview research: participant-level analysis and downloadable evidence log.
- Cross-interview synthesis: synthesis from analyses in the current session only.
- Golden-set evaluation: runs the selected engine against the five supplied hand-labelled cases.

## API keys
The app accepts a key in the sidebar for the selected provider. For deployment, use Streamlit Secrets instead:
```toml
GEMINI_API_KEY = "your-key"
OPENAI_API_KEY = "your-key"
ANTHROPIC_API_KEY = "your-key"
```
Only configure keys for providers you intend to use. Never commit secrets to GitHub. API calls may incur charges; check provider pricing and free-tier availability.

## Evidence and research integrity
- The included corpus is illustrative and its URLs are example placeholders, not verified evidence.
- The included golden set has five manually labelled examples; independently review labels and grow it to at least 20 diverse cases before reporting an evaluation result.
- Interview analyses and synthesis are session-based. Download the evidence log and synthesis outputs after each session.
- Do not enter names, phone numbers, account numbers, or other identifying/financial details. Anonymize notes before sending them to an external AI provider.
- Simulated interviews must be labelled as simulated and must not be reported as real user interviews.
- AI-generated findings require human review. A golden-set pass is not proof of general accuracy.
- This tool does not recommend funds or stocks and does not provide investment advice.
