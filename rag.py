import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

from evaluate import run_evaluation

# -----------------------------------
# Load environment variables
# -----------------------------------
load_dotenv()
groq_client = Groq(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------
# Load embedding model
# -----------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# Create Chroma client
# -----------------------------------
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="lectures")

# -----------------------------------
# Load data
# -----------------------------------
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# -----------------------------------
# Chunk text
# -----------------------------------
def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# -----------------------------------
# Ingest data
# -----------------------------------
def ingest(file_path):
    text = load_data(file_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"id_{i}"]
        )

    print("✅ Embeddings stored successfully.")

# -----------------------------------
# Retrieve relevant chunks
# -----------------------------------
def retrieve(query, k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]

# -----------------------------------
# Plain LLM Answer (No RAG)
# -----------------------------------
def generate_plain_llm_answer(question):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer clearly and concisely."},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

# -----------------------------------
# RAG Answer
# -----------------------------------
def generate_rag_answer(question, context_docs):
    context = "\n".join(context_docs)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY using the provided context. If not found, say 'Not found in provided knowledge.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()
def retrieve(query, k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]
def generate_answer(question, context_docs):
    context = "\n".join(context_docs)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer ONLY using the provided context."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

# -----------------------------------
# MAIN PIPELINE
# -----------------------------------
if __name__ == "__main__":
    ingest("data/lectures.txt")

    while True:
        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("\n==============================")
        print("QUESTION:", question)
        print("==============================")

        # 1️⃣ Plain LLM
        plain_answer = generate_plain_llm_answer(question)

        print("\n🔹 Plain LLM Answer:\n")
        print(plain_answer)

        # 2️⃣ Retrieve context
        docs = retrieve(question)

        # 3️⃣ RAG Answer
        rag_answer = generate_rag_answer(question, docs)

        print("\n------------------------------")
        print("🔹 RAG Answer:\n")
        print(rag_answer)

        # 4️⃣ Evaluate Faithfulness (Auto comparison)
        plain_eval = run_evaluation(plain_answer, docs, rag_answer)
        rag_eval = run_evaluation(rag_answer, docs, rag_answer)

        plain_hallu = plain_eval["faithfulness"]["hallucination_score"]
        rag_hallu = rag_eval["faithfulness"]["hallucination_score"]

        improvement = plain_hallu - rag_hallu

        print("\n==============================")
        print("📊 Evaluation Metrics")
        print("==============================")
        print("Plain LLM Hallucination Score:", plain_hallu)
        print("RAG Hallucination Score:", rag_hallu)
        print("Hallucination Reduction:", round(improvement, 3))
        print("Semantic Similarity (Plain vs RAG):",
              plain_eval["semantic_similarity"])
        print("==============================\n")