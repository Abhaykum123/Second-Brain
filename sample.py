import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=api_key,
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer questions clearly using simple words."
    ),
    (
        "human",
        "{question}"
    )
])

# LangChain chain
chain = prompt | llm


def ask_question(question: str):
    response = chain.invoke({
        "question": question
    })

    return response.content


if __name__ == "__main__":

    question = "Wnat is the capital of india"

    try:
        answer = ask_question(question)

        print("\nAnswer:")
        print(answer)

    except Exception as e:

        print("\nError:")
        print(e)
