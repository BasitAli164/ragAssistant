# 📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions about its content.

The application extracts text from the uploaded PDF, splits the document into token-aware chunks, generates semantic embeddings, stores them in a **FAISS vector index**, retrieves the most relevant chunks for a user's question, and uses **Groq's GPT-OSS 120B** model to generate a grounded answer.

The application is built with **Python and Streamlit** and can be deployed on **Streamlit Community Cloud**.

---

## 🚀 Features

* 📄 Upload PDF documents directly through the web interface
* 🔎 Extract text from PDF files using `pypdf`
* 🔤 Token-aware document chunking
* ✂️ Overlapping chunks for better retrieval
* 🧠 Generate semantic embeddings using Sentence Transformers
* 🗃️ Store embeddings in a local FAISS vector index
* 🔍 Perform semantic similarity search
* 🤖 Generate answers using Groq's `openai/gpt-oss-120b`
* 📌 Display the retrieved source chunks
* ⚡ Fast inference through Groq
* 🌐 Streamlit-based interactive interface
* ☁️ Deployable on Streamlit Community Cloud
* 🔐 API key handled through Streamlit Secrets

---

## 🏗️ RAG Architecture

The application follows this pipeline:

```text
                         PDF Document
                              │
                              ▼
                    ┌──────────────────┐
                    │  PDF Extraction  │
                    │      pypdf       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Text Cleaning   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Tokenization &   │
                    │    Chunking      │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌───────────────────────┐
                  │ Sentence Transformers│
                  │     Embeddings        │
                  └───────────┬───────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │      FAISS       │
                    │   Vector Index   │
                    └────────┬─────────┘
                             │
                             │
                      User Question
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Question Embedding  │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Semantic Search  │
                    │      FAISS       │
                    └────────┬─────────┘
                             │
                     Top-K Relevant
                         Chunks
                             │
                             ▼
                    ┌──────────────────┐
                    │     Groq API     │
                    │  GPT-OSS 120B    │
                    └────────┬─────────┘
                             │
                             ▼
                       Final Answer
```

---

## 🧠 How RAG Works in This Application

Instead of sending the entire PDF to the language model, the application first creates a searchable vector representation of the document.

### Document Ingestion

When a PDF is uploaded:

```text
PDF
 ↓
Text Extraction
 ↓
Text Cleaning
 ↓
Tokenization
 ↓
Chunking
 ↓
Embedding Generation
 ↓
FAISS Index
```

### Question Answering

When the user asks a question:

```text
User Question
 ↓
Question Embedding
 ↓
FAISS Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Retrieved Context + Question
 ↓
Groq GPT-OSS 120B
 ↓
Answer
```

This allows the language model to answer questions using relevant sections of the uploaded document instead of processing the entire document every time.

---

## 🛠️ Technology Stack

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming language |
| Streamlit             | Web interface             |
| pypdf                 | PDF text extraction       |
| Sentence Transformers | Semantic embeddings       |
| FAISS                 | Vector similarity search  |
| NumPy                 | Numerical operations      |
| Groq                  | LLM inference             |
| GPT-OSS 120B          | Generative language model |

---

## 🤖 Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts document chunks and user questions into dense vector representations.

These vectors allow the application to perform semantic similarity search.

### Language Model

```text
openai/gpt-oss-120b
```

The model is accessed through the Groq API and is responsible for generating the final answer based on the retrieved document context.

---

## 📦 Project Structure

```text
pdf-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-assistant.git
```

Move into the project directory:

```bash
cd pdf-rag-assistant
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure Groq API Key

The application requires a Groq API key.

### Local Development

Create the following file:

```text
.streamlit/
└── secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Do not commit `secrets.toml` to GitHub.

The `.gitignore` file should contain:

```gitignore
.streamlit/secrets.toml
.env
venv/
.venv/
__pycache__/
*.pyc
```

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Typically:

```text
http://localhost:8501
```

---

## 📄 Using the Application

### Step 1 — Upload a PDF

Upload a text-based PDF document through the interface.

Examples include:

* Research papers
* Books
* Reports
* Lecture notes
* Technical documentation
* Academic documents
* Project documentation

### Step 2 — Process the PDF

Click:

```text
🚀 Process PDF
```

The application will:

1. Extract the PDF text
2. Clean the text
3. Tokenize the document
4. Create overlapping chunks
5. Generate embeddings
6. Build a FAISS vector index

### Step 3 — Ask a Question

Enter a question such as:

```text
What are the main findings of this paper?
```

or:

```text
What methodology was used in the study?
```

or:

```text
What are the limitations mentioned by the authors?
```

### Step 4 — Retrieve Context

FAISS searches the vector index and retrieves the most semantically relevant chunks.

The default configuration retrieves:

```text
Top-K = 5
```

### Step 5 — Generate the Answer

The retrieved context is sent to:

```text
Groq → GPT-OSS 120B
```

The model generates an answer based on the retrieved document context.

---

## ⚙️ RAG Configuration

The current configuration is:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 350

CHUNK_OVERLAP = 50

TOP_K = 5
```

### Chunk Size

```text
350 tokens
```

Each document chunk contains approximately 350 tokens.

### Chunk Overlap

```text
50 tokens
```

Consecutive chunks overlap by 50 tokens to reduce the chance of losing important information at chunk boundaries.

### Top-K Retrieval

```text
5 chunks
```

The application retrieves the five most relevant chunks for each question.

---

## 🗃️ Vector Database

This project uses:

```text
FAISS
```

FAISS is used for efficient similarity search over dense vector embeddings.

The application uses:

```python
faiss.IndexFlatIP
```

with normalized embeddings.

Because the embeddings are normalized, inner product similarity corresponds to cosine similarity.

---

## 🔐 Security

The Groq API key should never be hard-coded into the source code.

❌ Do not do this:

```python
client = Groq(
    api_key="gsk_xxxxxxxxx"
)
```

Instead, use Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The secret file should not be committed to GitHub.

---

## ☁️ Deploy on Streamlit Community Cloud

### 1. Push the project to GitHub

Your repository should contain:

```text
app.py
requirements.txt
README.md
.gitignore
```

### 2. Open Streamlit Community Cloud

Create a new application and select your GitHub repository.

Set:

```text
Main file path:
app.py
```

### 3. Add the Groq Secret

In the application's Secrets configuration, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Deploy

Streamlit will install the dependencies from:

```text
requirements.txt
```

and launch:

```text
app.py
```

---

## ⚠️ Current Limitations

This version intentionally keeps the architecture simple.

### 1. Text-Based PDFs Only

The current PDF extraction uses `pypdf`.

Scanned PDFs or image-only PDFs require OCR and are not supported by the current version.

For example:

```text
Scanned PDF
      ↓
Image
      ↓
No machine-readable text
      ↓
pypdf cannot extract the actual text
```

OCR can be added in a future version.

---

### 2. FAISS Index Is Session-Based

The FAISS index is stored in Streamlit session state.

Therefore, the current application is designed primarily for:

* Learning
* Demonstration
* Portfolio projects
* Small-scale document Q&A

The vector index is not currently persisted in a permanent external database.

---

### 3. One Document at a Time

The current interface processes one uploaded PDF at a time.

Multi-document RAG can be added later by maintaining document metadata and a persistent vector store.

---

### 4. No Conversation Memory

Each question is treated independently.

The current version does not maintain a multi-turn conversation history.

Conversation memory can be added as a future enhancement.

---

## 🔮 Future Improvements

Potential improvements include:

* 🔤 OCR support for scanned PDFs
* 📚 Multi-document RAG
* 💾 Persistent FAISS indexes
* 🗄️ Metadata storage
* 📄 Page-level source citations
* 🔍 Hybrid keyword + semantic retrieval
* 🎯 Reranking retrieved chunks
* 💬 Conversation memory
* 🧠 Query expansion
* 📊 Retrieval evaluation
* 🛡️ Prompt-injection detection
* 📈 RAG evaluation using metrics such as faithfulness and retrieval relevance
* 👥 Multi-user document isolation
* 🔐 Authentication
* ⚡ Streaming model responses
* 🗃️ Production vector databases such as Qdrant or PostgreSQL with pgvector

---

## 📊 Example RAG Pipeline

For a research paper:

```text
Research Paper
      │
      ▼
Extract Text
      │
      ▼
Create ~350-token chunks
      │
      ▼
Generate embeddings
      │
      ▼
Store in FAISS
      │
      │
      │       User:
      │       "What methodology
      │        did the researchers use?"
      │
      ▼
Embed Question
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top 5 Relevant Chunks
      │
      ▼
Groq GPT-OSS 120B
      │
      ▼
Grounded Answer
```

---

## 🎯 Project Objective

The main objective of this project is to demonstrate the practical implementation of a Retrieval-Augmented Generation system using open-source/local embedding technology, FAISS vector search, and a modern large language model accessed through Groq.

The project demonstrates the complete RAG pipeline rather than relying on a high-level RAG framework:

```text
Document Processing
        +
Embedding Generation
        +
Vector Search
        +
Context Retrieval
        +
LLM Generation
        =
Retrieval-Augmented Generation
```

---

## 📜 License

This project is intended for educational, research, and portfolio purposes.

Add an appropriate open-source license to the repository if you plan to distribute or modify the project publicly.

---

## 👨‍💻 Author

**Basit Ali**

Computer Science Student
AI Engineer & Full-Stack Developer
Generative AI • RAG • AI Agents • Full-Stack Development

GitHub: `BasitAli164`

---

## ⭐ If You Find This Project Useful

If this project helps you learn about Retrieval-Augmented Generation, consider giving the repository a ⭐ on GitHub.
