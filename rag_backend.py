"""
rag_backend.py  -  BACKEND (AI logic)
LangChain RAG pipeline with conversational memory:
Groq LLM + ChromaDB (local) + HuggingFace all-MiniLM-L6-v2 embeddings.
Two "modes" share the same model and database but use different system prompts:
  general -> normal chatbot (index.html)
  pcmc    -> Pimpri-Chinchwad guide (map.html).  No fine-tuning, just a prompt + data/.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

BASE_DIR = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")
DATA_DIR = BASE_DIR / "data"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Each system prompt must contain 
SYSTEM_PROMPTS = {
    "general": (
        "You are a friendly local guide for Pimpri-Chinchwad (PCMC), in Pune district, "
        "Maharashtra, India. Answer questions about its areas, landmarks, history, "
        "industry, transport, civic information and daily life. "
        "Use the retrieved context first; if it does not cover the question, you may use "
        "your general knowledge but say clearly when you are unsure. "
        "Never invent addresses, phone numbers, timings, prices or routes. "
        "Use the earlier conversation to understand follow-up questions. "
        "If the question is not about Pimpri-Chinchwad or nearby Pune, politely steer "
        "the user back to the city. Mention area names (like Nigdi, Akurdi, Wakad) "
        "when relevant. Keep answers within five sentences."
        "\n\n{context}"
    ),
    "pcmc": (
        "You are a friendly local guide for Pimpri-Chinchwad (PCMC), in Pune district, "
        "Maharashtra, India. Answer questions about its areas, landmarks, history, "
        "industry, transport, civic information and daily life. "
        "Use the retrieved context first; if it does not cover the question, you may use "
        "your general knowledge but say clearly when you are unsure. "
        "Never invent addresses, phone numbers, timings, prices or routes. "
        "Use the earlier conversation to understand follow-up questions. "
        "If the question is not about Pimpri-Chinchwad or nearby Pune, politely steer "
        "the user back to the city. Mention area names (like Nigdi, Akurdi, Wakad) "
        "when relevant. Keep answers within five sentences."
        "\n\n{context}"
    ),
}

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# In-memory chat history per session (resets when the server restarts)
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def _load_documents():
    """Load .txt / .md files from the data/ folder."""
    docs = []
    if DATA_DIR.exists():
        for pattern in ("**/*.txt", "**/*.md"):
            docs.extend(
                DirectoryLoader(
                    str(DATA_DIR), glob=pattern, loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                ).load()
            )
    return docs


def _build_retriever(embeddings):
    """Return a retriever over data/ files, or None if there are no documents."""
    docs = _load_documents()
    if not docs:
        return None
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="rag_docs",
    )
    # Index only on first run; later runs reuse the saved database
    if vectorstore._collection.count() == 0:
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        vectorstore.add_documents(splits)
        print(f"[info] indexed {len(splits)} chunks")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def _make_chain(llm, retriever, system_text: str):
    if retriever is None:
        # No documents: plain conversation that answers from chat history only
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text.replace("{context}", "")),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | llm | StrOutputParser() | RunnableLambda(lambda t: {"answer": t})
    else:
        # 1) Rewrite follow-up questions into standalone ones using chat history
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )
        # 2) Answer using retrieved context + chat history
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        chain = create_retrieval_chain(
            history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt)
        )

    # 3) Automatic per-session memory
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def build_chains() -> dict:
    """Build one conversational chain per mode (model + vector DB are shared)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Put it in your .env file.")

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    retriever = _build_retriever(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    if retriever is None:
        print("[info] data/ is empty -> running in history-only mode")
    return {mode: _make_chain(llm, retriever, text) for mode, text in SYSTEM_PROMPTS.items()}


def ask(chains: dict, mode: str, question: str, session_id: str) -> dict:
    chain = chains.get(mode, chains["general"])
    result = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    sources = sorted(
        {Path(d.metadata["source"]).name for d in result.get("context", []) if d.metadata.get("source")}
    )
    return {"answer": result["answer"], "sources": sources}


def get_history(session_id: str) -> list[dict]:
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in get_session_history(session_id).messages
    ]


def clear_history(session_id: str) -> None:
    _store.pop(session_id, None)
