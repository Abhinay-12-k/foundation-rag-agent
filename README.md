Foundation RAG Agent :

This project demonstrates a Retrieval-Augmented Generation (RAG) system combined with evaluation metrics and a simple reflection agent. It compares pure LLM answers to RAG-grounded answers and evaluates key metrics.

Features :

RAG Pipeline: Retrieves relevant chunks from a local knowledge base.

LLM vs. RAG Answers: Generates responses from both a standalone LLM and a RAG-enhanced context.

Evaluation: Measures retrieval recall, hallucination (faithfulness), exact match, and semantic similarity.

How to Use

Ingest Data: The script ingests text chunks into ChromaDB.

Ask Questions: You can ask questions in a loop.

Compare Answers: It generates an LLM-only answer and a RAG answer.

Evaluate: If you provide a ground truth, it evaluates the response based on multiple metrics.

Files

rag.py: The main RAG pipeline.

evaluate.py: Evaluation functions for measuring accuracy and hallucination.

agent.py: A simple reflection agent (if implemented).

lectures.txt: Sample knowledge base text.

requirements.txt: Dependencies.

Dependencies :

Install dependencies via:

pip install -r requirements.txt
Example Usage
python rag.py

Follow the prompts to ask questions and get RAG-based answers.
