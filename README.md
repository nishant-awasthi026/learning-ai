# Learning AI

A collection of AI and LLM experiments, implementations, and mini-projects exploring various LLM techniques, agents, APIs, and frameworks.

## Projects & Modules

1. **LLM API CALL**
   - Basics of calling and interfacing with LLM APIs (e.g. Groq).

2. **PYDANTIC_JSON**
   - Structured JSON output parsing and validation using Pydantic schemas with LLMs.

3. **ReAct Chaining**
   - Implementation of Reasoning and Acting (ReAct) agent workflow patterns.

4. **STREAM LLM API**
   - Real-time token streaming with LLM APIs for interactive chat and CLI streaming.

5. **PERSONAL CHAT**
   - Interactive resume AI chatbot built with FastAPI and Groq API.
   - Includes custom web frontend (`index.html`), API endpoints, and resume parsing.


## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or `pip`
- Groq API Key

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/nishant-awasthi026/learning-ai.git
   cd learning-ai
   ```

2. Navigate to the project of your choice, configure `.env` by copying `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Add your API keys to `.env`.

3. Install dependencies and run:
   ```bash
   uv sync
   # or pip install -r requirements.txt
   ```
