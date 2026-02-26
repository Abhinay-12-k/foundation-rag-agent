import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1️⃣ Retrieval Evaluation
# -----------------------------

def evaluate_retrieval(retrieved_docs, ground_truth):
    hits = 0

    for doc in retrieved_docs:
        if ground_truth.lower() in doc.lower():
            hits += 1

    recall = hits / len(retrieved_docs) if retrieved_docs else 0

    return {
        "retrieval_recall": round(recall, 3),
        "hits": hits
    }


# -----------------------------
# 2️⃣ Faithfulness / Hallucination Check
# -----------------------------
def evaluate_faithfulness(answer, retrieved_docs):
    context = " ".join(retrieved_docs).lower()

    sentences = re.split(r'[.?!]', answer)
    unsupported = []

    for sentence in sentences:
        sentence = sentence.strip().lower()
        if sentence and sentence not in context:
            unsupported.append(sentence)

    total_sentences = len([s for s in sentences if s.strip()])

    hallucination_score = (
        len(unsupported) / total_sentences if total_sentences > 0 else 0
    )

    return {
        "hallucination_score": round(hallucination_score, 3),
        "unsupported_sentences": unsupported
    }


# -----------------------------
# 3️⃣ Exact Match
# -----------------------------
def evaluate_exact_match(answer, ground_truth):
    return answer.strip().lower() == ground_truth.strip().lower()


# -----------------------------
# 4️⃣ Semantic Similarity
# -----------------------------
def evaluate_semantic_similarity(answer, ground_truth):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb1 = model.encode([answer])
    emb2 = model.encode([ground_truth])

    similarity = cosine_similarity(emb1, emb2)[0][0]

    return round(float(similarity), 3)


# -----------------------------
# 5️⃣ Combined Evaluation
# -----------------------------
def run_evaluation(answer, retrieved_docs, ground_truth):
    retrieval_metrics = evaluate_retrieval(retrieved_docs, ground_truth)
    faithfulness_metrics = evaluate_faithfulness(answer, retrieved_docs)
    exact_match = evaluate_exact_match(answer, ground_truth)
    semantic_score = evaluate_semantic_similarity(answer, ground_truth)

    return {
        "retrieval": retrieval_metrics,
        "faithfulness": faithfulness_metrics,
        "exact_match": exact_match,
        "semantic_similarity": semantic_score
    }
# -----------------------------
# 6️⃣ Test Block
# -----------------------------
if __name__ == "__main__":

    # Dummy test data
    answer = "Self-supervised learning is learning from raw data without labels."
    retrieved_docs = [
        "Self-supervised learning allows models to learn from raw data without human labels.",
        "Foundation models are trained using self-supervised learning."
    ]
    ground_truth = "Self-supervised learning is learning from raw data without labels."

    results = run_evaluation(answer, retrieved_docs, ground_truth)

    print("\nEvaluation Results:\n")
    print(results)