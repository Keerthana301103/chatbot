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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@/chatbotdb?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_ID")
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# Define Document Model
class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(Text, nullable=False, unique=True)
    content = Column(Text, nullable=False)
    uploaded_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

# Ensure tables exist
Base.metadata.create_all(engine)

# Initialize Streamlit
st.title("PDF Chatbot")
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize LLM and Vector DB lazily
llm = Ollama(model="llama3.2")
vector_db = None

if uploaded_file:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    
    with st.spinner("Processing document..."):
        try:
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() or "" for page in pdf_reader.pages]).strip()

            if not content:
                st.sidebar.error("No extractable text found in the document.")
            else:
                # Store document in DB (avoid duplicate uploads)
                with SessionLocal() as db:
                    existing_doc = db.query(Document).filter(Document.filename == uploaded_file.name).first()
                    if not existing_doc:
                        doc = Document(filename=uploaded_file.name, content=content)
                        db.add(doc)
                        db.commit()
                    else:
                        st.sidebar.warning("This document is already uploaded.")

                # Split text for embeddings
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                texts = text_splitter.split_text(content)

                # Create embeddings and store in FAISS
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                vector_db = FAISS.from_texts(texts, embeddings)

                st.sidebar.success("Document processed successfully!")

        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")

# Chatbot Section
st.header("Ask Questions")
user_question = st.text_input("Enter your question:")

if user_question:
    if vector_db:
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
    else:
        st.error("No documents available. Please upload a PDF first.")

# Streamlit Cloud Run Compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    st.run()
