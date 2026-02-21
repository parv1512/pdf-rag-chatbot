# Intelligent PDF RAG Chatbot
Local, private question-answering over your PDFs using only open-source tools.

![Demo](images/screenshot-1.png)

Sample PDFs (publicly available)
- https://www.defence.lk/upload/ebooks/A%20History%20of%20India.pdf
- https://arxiv.org/pdf/1706.03762.pdf

## Features
- 100% local (Ollama + Chroma)
- Multi-PDF support
- Page-aware source citations
- Adjustable model, temperature, top-k
- Clean chat UI with history

## Current Limitations
- The system cannot directly reference or retain context from previously sent messages.
- Performance may be slower when processing large files, resulting in increased response and processing time.

## Tech Stack
- LangChain (LCEL style)
- Ollama (llama3.2, mxbai-embed-large)
- Chroma vector store
- Streamlit

## Quick Start

```bash
# 1. Install Ollama models
ollama pull llama3.2
ollama pull mxbai-embed-large

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (Quick Tip  : don't forget to start Ollama before running this command)
streamlit run app.py

