# 🔍 Repo-Chat

A local RAG (Retrieval-Augmented Generation) application that ingests GitHub repositories and uses a **Coder + Critic Agentic Loop** to answer questions and suggest verified code fixes.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)

## ✨ Features

- **🧠 Syntax-Aware Ingestion**: Reads code by function/class, not just by line number
- **🔎 Semantic + Keyword Search**: Hybrid retrieval finds code by meaning AND exact matches
- **🤖 Multi-Agent Loop**: Coder generates fixes, Critic reviews against quality rules
- **🔒 Private & Local**: All data stays on your machine (ChromaDB + local embeddings)
- **🔄 Switchable LLM Providers**: Groq (free) or DeepSeek (high quality)

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   GitHub Repo   │────▶│   Ingest     │────▶│   ChromaDB      │
│                 │     │   Pipeline   │     │   (Vectors)     │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   User Query    │────▶│   Hybrid     │────▶│   Retrieved     │
│                 │     │   Search     │     │   Context       │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
                        ┌──────────────────────────────────────┐
                        │         Agentic Loop                 │
                        │  ┌────────┐      ┌────────┐          │
                        │  │ Coder  │─────▶│ Critic │──┐       │
                        │  │ Agent  │◀─────│ Agent  │  │       │
                        │  └────────┘      └────────┘  │       │
                        │       ▲                      │       │
                        │       └──────────────────────┘       │
                        └──────────────────────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────┐
                              │  Verified Fix   │
                              └─────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/repo-chat.git
cd repo-chat
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
GROQ_API_KEY="gsk_your_key_here"
# Optional: For DeepSeek provider
# DEEPSEEK_API_KEY="sk_your_key_here"
LLM_PROVIDER="GROQ"
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📖 Usage

1. **Ingest a Repository**: Paste a GitHub URL in the sidebar and click "Ingest Repository"
2. **Ask Questions**: Type questions about the codebase in the chat
3. **Get Fixes**: Ask for code fixes - the Coder generates, Critic reviews

### Example Queries

- "How does Flask handle route registration?"
- "Fix the `_get_padding_width` method to account for `pad_edge`"
- "What's the structure of the Config class?"

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain |
| LLM (Free) | Groq (llama-3.3-70b-versatile) |
| LLM (Quality) | DeepSeek (deepseek-coder) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| UI | Streamlit |

## 📁 Project Structure

```
repo-chat/
├── app.py              # Streamlit UI
├── agent.py            # Coder & Critic agents
├── ingest.py           # Repository ingestion pipeline
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔧 Configuration

### LLM Providers

Switch providers via the UI or `.env`:

- **GROQ** (Default): Free tier, fast, good for development
- **DEEPSEEK**: Higher quality code generation

### Critic Rules

Default rules enforced by the Critic Agent:
- All functions must have type hints
- No bare `except:` clauses
- Only use APIs shown in retrieved context
- Match original function signatures

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - feel free to use this project for learning and development.

---

Built with ❤️ using LangChain, Streamlit, and ChromaDB
