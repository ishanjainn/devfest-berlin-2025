# Trip Planner Agent with Observability

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Required Environment Variables

#### OpenAI API Key (Required)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```
- Get your key at: https://platform.openai.com/api-keys
- This is **required** for the agent to work

#### Serper API Key (Optional - for web search)
```bash
export SERPER_API_KEY="your-serper-api-key-here"
```
- Get a free key at: https://serper.dev/
- **Optional** - agent works without this but with less real-time information
- Free tier: 2,500 searches/month

#### Configure OpenTelemetry destination

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-me-central-1.grafana.net/otlp"

export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20xyz"
```

### 3. Run the Agent
```bash
openlit-instrument python trip_planner_agent.py
```

