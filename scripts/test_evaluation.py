from core.evaluation.retrieval_evaluator import RetrievalEvaluator

def main():

    evaluator = RetrievalEvaluator()

    # Dummy retrieved chunks
    retrieved = [
        (0.1, {"text": "FAISS is used for similarity search"}),
        (0.2, {"text": "Transformers use attention"}),
    ]

    relevant = ["FAISS is used for similarity search"]

    recall = evaluator.recall_at_k(retrieved, relevant)
    precision = evaluator.precision_at_k(retrieved, relevant)

    print(f"Recall@K: {recall}")
    print(f"Precision@K: {precision}")


if __name__ == "__main__":
    main()