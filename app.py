import streamlit as st
import os
import tempfile
import shutil
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Intelligent PDF Chatbot", layout="wide")
st.title("Intelligent PDF/RAG Chatbot")
st.caption("Upload textbooks, papers, manuals → ask anything. 100% local & private.")

#               SIDEBAR SETTINGS
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "LLM Model",
        ["llama3.2", "llama3.2:3b", "qwen2.5:7b", "gemma2:9b", "phi3.5"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Retrieve top K chunks", 2, 12, 5)

    st.divider()
    if st.button("🗑️ Clear Database & Chat", type="primary"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Database and chat cleared!")
        st.rerun()

#               SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

#               DOCUMENT UPLOAD & INDEXING
st.header("1. Upload Your PDFs")

uploaded_files = st.file_uploader(
    "Drag & drop one or more PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="You can upload multiple documents at once."
)

if uploaded_files and st.button("🚀 Process & Index Documents", type="primary"):
    with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}")
            finally:
                os.unlink(tmp_path)

        if not all_docs:
            st.error("No readable text found in the uploaded PDFs.")
        else:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                add_start_index=True,
            )
            splits = text_splitter.split_documents(all_docs)

            # Embed & store
            embeddings = OllamaEmbeddings(model="mxbai-embed-large")
            
            # Remove old index if exists (fresh start on new upload session)
           # if os.path.exists("./chroma_db"):
            #    shutil.rmtree("./chroma_db")

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )

            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Indexed **{len(splits)}** chunks from **{len(uploaded_files)}** document(s). Ready to chat!")

#               CHAT INTERFACE  
st.header("2. Ask Questions")

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("Please upload and process at least one PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # ── LLM & Retriever ─────────────────────────────────────
                llm = ChatOllama(model=model_name, temperature=temperature)
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )

                # ── Prompt ──────────────────────────────────────────────
                prompt_template = """You are a helpful assistant that answers questions based **only** on the provided context.
If the information is not in the context, say: "I don't have enough information in the documents to answer this."

Always be concise, accurate and polite.

Context:
{context}

Question: {question}

Answer:"""

                qa_prompt = PromptTemplate.from_template(prompt_template)

                # ── Format retrieved documents with source info ────────
                def format_docs(docs):
                    return "\n\n".join(
                        f"From {doc.metadata.get('source', 'unknown file').split('/')[-1]}, page {doc.metadata.get('page', '?')+1}:\n{doc.page_content.strip()}"
                        for doc in docs
                    )

                # ── LCEL Chain ──────────────────────────────────────────
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | qa_prompt
                    | llm
                    | StrOutputParser()
                )

                # Generate answer
                answer = rag_chain.invoke(prompt)

                # Show answer
                st.markdown(answer)

                # Retrieve sources for display (independent step)
                retrieved_docs = retriever.invoke(prompt)

                if retrieved_docs:
                    with st.expander("📑 Sources", expanded=False):
                        for i, doc in enumerate(retrieved_docs):
                            page = doc.metadata.get("page")
                            page_str = f"page {int(page) + 1}" if isinstance(page, (int, float)) else "no page info"
                            filename = doc.metadata.get("source", "Unknown").split("/")[-1]
                            st.markdown(f"**Source {i+1}** — *{filename}* ({page_str})")
                            preview = doc.page_content[:380].replace("\n", " ").strip()
                            if len(doc.page_content) > 380:
                                preview += "..."
                            st.caption(preview)

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Powered by Ollama + LangChain + Chroma • local RAG chatbot • 2026 portfolio project")