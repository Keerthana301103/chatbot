import streamlit as st
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import os

# Database Configuration (Using Streamlit Secrets)
DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define Document Model
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
    uploaded_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# Create Database Tables (if not exists)
Base.metadata.create_all(engine)

# Initialize Ollama LLM
llm = Ollama(model="llama3.2")

# Streamlit UI
st.title("PDF QnA Chatbot")

# Sidebar for File Upload
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

vector_db = None

if uploaded_file:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Processing document..."):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

            # Store document in the database (Prevent Duplicates)
            db = SessionLocal()
            existing_doc = db.query(Document).filter(Document.filename == uploaded_file.name).first()
            if not existing_doc:
                doc = Document(filename=uploaded_file.name, content=content)
                db.add(doc)
                db.commit()
            else:
                st.sidebar.warning("This document is already uploaded.")
            db.close()

            # Split text for embedding
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_text(content)

            # Create FAISS vector database
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_db = FAISS.from_texts(texts, embeddings)

            st.sidebar.success("Document processed successfully!")

        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")

# Chatbot Section
st.header("Ask Questions")

user_question = st.text_input("Enter your question:")

if user_question and vector_db:
    with st.spinner("Searching for answers..."):
        try:
            # Retrieve relevant context
            docs = vector_db.similarity_search(user_question, k=3)

            # Use LangChain QA Chain
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            answer = qa_chain.run(input_documents=docs, question=user_question)

            st.subheader("Response:")
            st.write(answer)

        except Exception as e:
            st.error(f"Error generating answer: {e}")

elif user_question:
    st.error("No documents available. Please upload a PDF first.")
