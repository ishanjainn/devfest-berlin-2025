# 🌟 Trip Planner Agent Setup Guide

## 📋 Quick Setup

### 1. Install Dependencies
```bash
pip install -r trip_planner_requirements.txt
```

### 2. Required API Keys

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

### 3. Run the Agent
```bash
python trip_planner_agent.py
```

## 🔧 Troubleshooting

### Serper Search Failures
If you see repeated "Failed Search the internet with Serper" messages:

1. **No Serper API Key**: The agent will automatically fallback to LLM-only mode
2. **Invalid Serper Key**: Check your API key is correct
3. **Rate Limits**: You may have exceeded the free tier limits

**Solution**: The agent will work perfectly without Serper - it just uses the LLM's built-in knowledge instead of real-time web search.

### Missing Dependencies
```bash
# If you get import errors:
pip install --upgrade crewai crewai-tools langchain-openai openai openlit
```

## 🎯 What Each Mode Does

### With Serper API (Enhanced Mode)
- ✅ Real-time web search for current information
- ✅ Latest attraction hours, prices, and availability
- ✅ Current weather and travel advisories
- ✅ Up-to-date local events and festivals

### Without Serper API (LLM-Only Mode)
- ✅ Comprehensive trip planning using AI knowledge
- ✅ Detailed itineraries and recommendations
- ✅ Budget analysis and travel tips
- ✅ Cultural insights and local experiences
- ⚠️ Information based on training data (may not be current)

## 💡 Tips

1. **Start Simple**: The agent works great without Serper for most trips
2. **Add Serper Later**: You can always add the Serper key later for enhanced searches
3. **Cost Management**: Monitor your OpenAI usage - the agent makes multiple API calls
4. **Save Results**: Each trip plan is automatically saved to a timestamped file

## 🔑 Environment Variables Summary

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (for web search)
export SERPER_API_KEY="..."

# Optional (for OpenLIT observability)
export OPENLIT_OTLP_ENDPOINT="http://localhost:4318"
```

That's it! The agent is designed to work smoothly with just the OpenAI key. 🚀
