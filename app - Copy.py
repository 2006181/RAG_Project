"""Streamlit interface for the book question-answering RAG project."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(page_title="Book Q&A", page_icon="📚", layout="centered")
st.title("📚 Ask your book")
st.caption("Upload a PDF book, then ask questions using its contents.")


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


@st.cache_resource
def get_llm() -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to the project's .env file."
        )
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's uploaded book. Use only the "
            "provided context. If the answer is not in the context, say: "
            '"I could not find the answer in the document."',
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


def index_pdf(pdf_bytes: bytes) -> Chroma:
    """Load and index an uploaded PDF using a temporary file for PyPDFLoader."""
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()
        if not pages:
            raise ValueError("No readable pages were found in this PDF.")

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        ).split_documents(pages)
        if not chunks:
            raise ValueError("No readable text was found in this PDF.")

        return Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
        )
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


uploaded_book = st.file_uploader("Upload a book (PDF)", type=["pdf"])

if uploaded_book is None:
    st.info("Choose a PDF to start. Your questions will be answered from that book.")
    st.stop()

pdf_bytes = uploaded_book.getvalue()
book_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("book_hash") != book_hash:
    try:
        with st.spinner("Reading and indexing the book…"):
            st.session_state.vectorstore = index_pdf(pdf_bytes)
        st.session_state.book_hash = book_hash
        st.session_state.book_name = uploaded_book.name
        st.session_state.messages = []
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

st.success(f"Ready: {st.session_state.book_name}")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about your book")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the book and preparing an answer…"):
                docs = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                ).invoke(question)
                context = "\n\n".join(doc.page_content for doc in docs)
                response = get_llm().invoke(
                    prompt.invoke({"context": context, "question": question})
                )
                answer = response.content
                sources = [
                    f"Page {doc.metadata.get('page', '?') + 1}"
                    if isinstance(doc.metadata.get("page"), int)
                    else f"Page {doc.metadata.get('page', '?')}"
                    for doc in docs
                ]
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        st.write(", ".join(dict.fromkeys(sources)))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": list(dict.fromkeys(sources))}
        )
    except Exception as exc:
        st.error(f"Could not answer this question: {exc}")
