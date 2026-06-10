import os
import time
import warnings
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

warnings.filterwarnings("ignore")
load_dotenv()

DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------
# Prompt
# ---------------------------------------------------

custom_prompt_template = """
You are a helpful assistant for question answering over a set of documents.

Rules:
1. If relevant information exists in the context, answer ONLY from the context.
2. If the context does not contain the answer, reply exactly:

Context is not available in data. Based on general knowledge, here is the answer:

and then answer from general knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"],
    )

# ---------------------------------------------------
# LLM (Local Ollama)
# ---------------------------------------------------

def load_llm():
    return ChatOllama(
        model="llama3.2",
    temperature=0.1,
    num_predict=150,
    )

# ---------------------------------------------------
# Embeddings
# ---------------------------------------------------

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------------------------------------
# Create FAISS Vector DB
# ---------------------------------------------------

def create_vector_db():
    print("Creating FAISS vector database...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embedder()

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print("FAISS database created successfully.")

# ---------------------------------------------------
# Build RAG Pipeline
# ---------------------------------------------------

def build_rag_chain(llm, prompt, db):

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "fetch_k": 5
        }
    )

    def format_docs(docs):

        print("\nRetrieved docs:", len(docs))

        for i, doc in enumerate(docs):

            print(f"\nDOC {i+1}")
            print(doc.page_content[:300])

        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def qa_bot():

    embeddings = get_embedder()

    try:
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("FAISS index loaded.")

    except Exception as e:

        print("FAISS load failed. Rebuilding index...")
        create_vector_db()

        db = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    llm = load_llm()
    prompt = set_custom_prompt()

    return build_rag_chain(llm, prompt, db)

# ---------------------------------------------------
# Load RAG once (IMPORTANT FIX)
# ---------------------------------------------------

print("Loading RAG system...")

qa_chain = qa_bot()

print("RAG system ready.")

# ---------------------------------------------------
# Ask Question
# ---------------------------------------------------

def final_result(query: str):

    answer_text = qa_chain.invoke(query)

    return answer_text

# ---------------------------------------------------
# Flask Routes
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("open_ai_trail.html")


@app.route("/ask", methods=["POST"])

def ask():

    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query parameter is required"}), 400

    query = data["query"].strip()

    # Greeting handling
    if query.lower() in ["hi", "hello", "hey", "hii"]:
        return jsonify({
            "result": "Hello! How can I help you today?"
        })

    start = time.time()

    answer_text = final_result(query)

    end = time.time()

    response_time = end - start

    response_time_line = f"⏱ Response time: {response_time:.2f} seconds\n"

    raw_response = response_time_line + answer_text

    formatted_response = "<br>".join(raw_response.split("\n"))

    return jsonify({"result": formatted_response})


@app.route("/reset", methods=["POST"])
def reset():

    return jsonify({
        "status": "ok",
        "message": "history cleared"
    }), 200

# ---------------------------------------------------
# Start Flask
# ---------------------------------------------------

if __name__ == "__main__":

    if not os.path.exists(DB_FAISS_PATH):
        create_vector_db()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8089
    )