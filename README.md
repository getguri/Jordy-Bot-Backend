# 🧠 Gen-AI-Backend: Enterprise NLP Processor & AI Core

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

A robust, high-performance backend processing engine built for advanced Natural Language Processing (NLP), semantic search pipelines, and orchestrating scalable generative AI microservices.

---

## ⚡ Core Capabilities

- **NLP Processing Pipelines:** Advanced text tokenization, chunking, and semantic transformation modules designed for high-throughput text analysis.
- **Vector Store Integration:** Optimized querying layers for vector databases to support low-latency similarity searches and retrieval-augmented workflows.
- **API Architecture:** Clean, asynchronous REST/gRPC endpoints designed to seamlessly integrate with frontend clients (React/Angular).
- **Extensible LLM Connectors:** Modular service wrappers supporting seamless integration with OpenAI, open-source LLMs, and embedding models.

---

## 💻 Tech Stack

* **Core Language:** Python 3.x, Asynchronous Programming (`asyncio`)
* **Frameworks:** FastAPI / Flask, LangChain, LlamaIndex
* **Data & Storage:** PostgreSQL, Chroma DB / Vector Stores
* **DevOps & Tooling:** Docker, Git, CI/CD Pipelines

---

## 🔄 System Architecture Flow

```text
[ Frontend / Client App ]
           │ (REST / JSON)
           ▼
[ FastAPI / API Gateway ]
           │
           ├─────────────────────────┐
           ▼                         ▼
[ NLP Processing Engine ]    [ Vector DB / Embeddings ]
 (Chunking & Tokenization)    (Semantic Similarity Search)
           │                         │
           └───────────┬─────────────H
                       ▼
         [ LLM Orchestration & Response ]




Push an existing Git repository

cd existing_repo
git remote rename origin old-origin
git remote add origin git@gitlab.hcl-edtech.com:eboxdev/genai-sandbox-backend.git
git push -u origin --all
git push -u origin --tags

Create a new repository

git clone git@gitlab.hcl-edtech.com:eboxdev/genai-sandbox-backend.git
cd genai-sandbox-backend
git switch -c main
touch README.md
git add README.md
git commit -m "add README"
git push -u origin main



