import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("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")
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


# 1. Data Preparation: load & chunk documents
def load_documents(data_dir: str) -> List[Dict]:
    docs = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith((".txt", ".md")):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": fname, "text": text})
    return docs


def chunk_documents(
    docs: List[Dict], chunk_size: int = 600, chunk_overlap: int = 100
) -> List[Dict]:
    """
    Chunk size ~600 chars with 100 overlap:

    - Big enough to keep a full policy clause/section together
    - Small enough that multiple chunks fit in the LLM context
    - Overlap avoids cutting important sentences in half
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "page_content": chunk,
                    "metadata": {"source": doc["source"], "chunk_id": i},
                }
            )
    return all_chunks


# 2. RAG Pipeline: embeddings + vector store
def build_vector_store(chunks: List[Dict]) -> Chroma:
    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


# 3. Prompt Engineering: initial and improved prompts

SYSTEM_PROMPT_V1 = """
You are a helpful company policy assistant.

Rules:
- Use the provided policy context to answer user questions.
- If the answer is not clearly in the context, you may say you are not sure.
- Be polite and concise.
- Prefer paraphrasing the policy in simple language.

Format:
Answer:
- ...

(Optional) Policy reference:
- Source file names if relevant.
"""

SYSTEM_PROMPT_V2 = """
You are a strict, compliance-focused company policy assistant.

Instructions:
- Answer ONLY using the context below.
- If information is missing or unclear, explicitly state:
  "This is not specified in the available policy documents."
- Do NOT guess or add any information that is not grounded in the context.
- Be concise and structured.
- Include a short justification referencing relevant snippets and file names.

Output format (Markdown):

## Answer
- ...

## Evidence
- Source: <file>, brief quote or summary

## Confidence
- One of: High / Medium / Low (based on how directly the context supports the answer).
"""


def build_user_prompt(question: str, context_chunks: List[Dict]) -> str:
    context_strs = []
    for c in context_chunks:
        meta = c["metadata"]
        context_strs.append(
            f"[SOURCE: {meta.get('source')} | CHUNK: {meta.get('chunk_id')}]\n{c['page_content']}"
        )
    context_block = "\n\n---\n\n".join(context_strs)
    user_prompt = f"""You are given company policy context and a user question.

Context:
{context_block}

User question:
{question}
"""
    return user_prompt


# 4. Ask LLM with retrieved context
def ask_llm(
    question: str,
    vectordb: Chroma,
    system_prompt: str = SYSTEM_PROMPT_V2,
    k: int = 4,
) -> str:
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return (
            "I could not find any relevant policy information for this question "
            "in the available documents."
        )

    context_chunks = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in docs
    ]
    user_prompt = build_user_prompt(question, context_chunks)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content


# 5. Simple CLI
def main():
    if not os.path.exists(CHROMA_DIR):
        print("Building vector store from data/ ...")
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        build_vector_store(chunks)

    vectordb = load_vector_store()
    print("Policy RAG assistant ready. Type 'exit' to quit.")
    while True:
        q = input("\nUser: ")
        if q.lower().strip() in ("exit", "quit"):
            break
        answer = ask_llm(q, vectordb)
        print("\nAssistant:\n", answer)


if __name__ == "__main__":
    main()
S
