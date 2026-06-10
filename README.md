# PDF RAG Chatbot using LangChain and Ollama

A Retrieval-Augmented Generation (RAG) chatbot that answers user queries using information retrieved from PDF documents.

## Features

* PDF document ingestion
* Text chunking using LangChain
* Semantic search with FAISS
* HuggingFace sentence embeddings
* Local LLM inference using Ollama (Llama 3.2)
* Context-aware question answering
* Flask-based web interface

## Tech Stack

* Python
* Flask
* LangChain
* FAISS
* HuggingFace Embeddings
* Ollama
* Llama 3.2

## Project Workflow

PDF Documents

↓

Text Chunking

↓

Embeddings Generation

↓

FAISS Vector Store

↓

Retriever

↓

Llama 3.2 (Ollama)

↓

Answer Generation

## Installation

Clone the repository:

```bash
git clone https://github.com/Rekhadumpa/rag-chatbot-using-llama.git
cd rag-chatbot-using-llama
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Ollama:

```bash
ollama run llama3.2
```

Start the application:

```bash
python app.py
```

## Usage

1. Add PDF documents to the `data` folder.
2. Build the vector database.
3. Start the Flask application.
4. Ask questions through the chatbot interface.

## Future Improvements

* Chat history support
* Multiple document collections
* Better UI/UX
* User authentication
* Deployment support

## Author

Rekha Dumpa
