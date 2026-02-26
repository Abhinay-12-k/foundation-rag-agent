from rag import retrieve, generate_answer
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_client = Groq(api_key=os.getenv("OPENAI_API_KEY"))


def reflect_on_answer(question, answer, context_docs):
    context = "\n".join(context_docs)

    reflection_prompt = f"""
You are a strict evaluator.

Context:
{context}

Question:
{question}

Answer:
{answer}

Is the answer fully supported by the context?
If yes, say: SUPPORTED
If not, say: NOT SUPPORTED and explain why briefly.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": reflection_prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def reflection_agent(question):
    # Step 1: Retrieve
    docs = retrieve(question)

    # Step 2: Initial Answer
    initial_answer = generate_answer(question, docs)

    # Step 3: Reflect
    reflection = reflect_on_answer(question, initial_answer, docs)

    # Step 4: Improve if needed
    if "NOT SUPPORTED" in reflection:
        improved_prompt = f"""
Use ONLY the provided context strictly.
If information is missing, say 'Not found in provided knowledge.'

Context:
{chr(10).join(docs)}

Question:
{question}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": improved_prompt}],
            temperature=0
        )

        final_answer = response.choices[0].message.content.strip()
    else:
        final_answer = initial_answer

    return {
        "retrieved_docs": docs,
        "initial_answer": initial_answer,
        "reflection": reflection,
        "final_answer": final_answer
    }


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (type 'exit'): ")

        if q.lower() == "exit":
            break

        result = reflection_agent(q)

        print("\n--- Retrieved Context ---")
        for d in result["retrieved_docs"]:
            print("-", d[:120], "...")

        print("\n--- Initial Answer ---")
        print(result["initial_answer"])

        print("\n--- Reflection ---")
        print(result["reflection"])

        print("\n--- Final Answer ---")
        print(result["final_answer"])