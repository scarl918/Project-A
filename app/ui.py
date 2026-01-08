import streamlit as st

from ingest import load_arxiv_docs
from chunker import chunk_documents
from embedder import Embedder
from vectorstore import FaissVectorStore
from rag import answer_query
from citations import format_citation
from summarizer import summarize_chunks

DATA_PATH = "data/arxiv_sample.jsonl"


@st.cache_resource
def build_backend():
    docs = load_arxiv_docs(DATA_PATH)
    chunks = chunk_documents(docs)

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    vectorstore = FaissVectorStore(dim=embeddings.shape[1])
    vectorstore.add(embeddings, chunks)

    return embedder, vectorstore, chunks


st.set_page_config(page_title="RAG Research Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG-Powered Research Assistant")
st.caption("Ask questions, get grounded answers with citations, or summarize papers.")

embedder, vectorstore, all_chunks = build_backend()

tab1, tab2 = st.tabs(["🔍 Ask a Question", "📝 Summarize a Paper"])

# -------------------- TAB 1: QA --------------------
with tab1:
    query = st.text_input("Ask a research question")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = answer_query(query, embedder, vectorstore)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        seen = set()
        idx = 1
        for s in sources:
            pid = s["meta"]["id"]
            if pid in seen:
                continue
            seen.add(pid)
            st.write(format_citation(s["meta"], idx))
            idx += 1


# -------------------- TAB 2: Summarization --------------------
with tab2:
    paper_id = st.text_input("Enter arXiv paper ID (e.g. 0704.0050)")

    if paper_id:
        paper_chunks = [c for c in all_chunks if c["meta"]["id"] == paper_id]

        if not paper_chunks:
            st.warning("Paper not found in index.")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_chunks(paper_chunks)

            st.subheader("Summary")
            st.write(summary)
