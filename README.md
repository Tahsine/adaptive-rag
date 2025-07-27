# Adaptive RAG – Hybrid and Intelligent Question-Answering System

**Adaptive RAG** (Retrieval-Augmented Generation) is a scalable pipeline that combines local vector retrieval and real-time web search. Its goal is to dynamically optimize the information retrieval strategy for each question, ensuring reliable, relevant, and non-hallucinated answers.

- Read more in my [blog](https://reality-checkpoint.vercel.app/blog/article-adaptive-rag-the-next-step-for-trylly-intelligent-llms)

## 🚀 Utilisation

1. Clone the repository and activate a virtual environment:
   Make sure to install uv before using it. Look for the installation guide at the [official site](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tahsine/adaptive-rag.git
cd adaptive-rag
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):

```bash
cp .env.example .env
```

4. Launch the assistant locally to test with some questions:

```bash
# Run the ingest_web_data.py first to upload some data from the web

uv run src/ingest_web_data.py

# You can edit the question in the main.py file
uv run main.py
```

## Architecture Idea

![Adaptive RAG Architecture](/public/image/adaptive-rag.png)

<div align="center">

_Source: [LangChain](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)_

</div>
