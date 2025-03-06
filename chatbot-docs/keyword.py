import streamlit as st
import PyPDF2
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain.docstore.document import Document as LC_Document

# Load environment variables
load_dotenv(r'C:\Users\s.anumandla\Desktop\QnA chatbot\.env')

# Database Configuration
DATABASE_URL = "postgresql://postgres:1234@localhost:5432/chatbotdb"
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

# Create Database Tables
Base.metadata.create_all(engine)

# Initialize Ollama LLM
llm = Ollama(model="llama3.2") 

# Streamlit UI
st.title("PDF Chatbot (Improved Keyword-Based)")

# Sidebar for File Upload
st.sidebar.header("Upload a PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    
    with st.spinner("Processing document..."):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text() is not None])

            # Store document in the database (prevent duplicates)
            db = SessionLocal()
            existing_doc = db.query(Document).filter(Document.filename == uploaded_file.name).first()
            if not existing_doc:
                doc = Document(filename=uploaded_file.name, content=content)
                db.add(doc)
                db.commit()
            else:
                st.sidebar.warning("This document is already uploaded.")
            db.close()

            st.sidebar.success("Document processed successfully!")

        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")

# Function to perform TF-IDF based keyword search
def keyword_search_tfidf(question, texts, num_results=3):
    """Search for relevant document sections using TF-IDF."""
    if not texts:
        return []
    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts + [question])
    
    query_vector = tfidf_matrix[-1]  # The last row is the question vector
    scores = (tfidf_matrix[:-1] * query_vector.T).toarray().flatten()  # Compute similarity

    # Get top N matching sentences
    top_indices = np.argsort(scores)[::-1][:num_results]
    return [texts[i] for i in top_indices if scores[i] > 0]  # Filter out zero-score results

# Chatbot Section
st.header("Ask Questions")

# User Input
user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Searching for answers..."):
        try:
            # Retrieve all document contents from the database
            db = SessionLocal()
            docs = db.query(Document).all()
            db.close()

            if docs:
                all_texts = [doc.content for doc in docs]
                relevant_texts = keyword_search_tfidf(user_question, all_texts, num_results=3)
                
                if relevant_texts:
                    # Convert relevant texts into LangChain document format
                    lc_documents = [LC_Document(page_content=text) for text in relevant_texts]

                    # Use LangChain QA Chain to generate response
                    qa_chain = load_qa_chain(llm, chain_type="stuff")
                    answer = qa_chain.run(input_documents=lc_documents, question=user_question)

                    st.subheader("Response:")
                    st.write(answer)
                else:
                    st.warning("No matching content found in the document.")

            else:
                st.error("No documents available. Please upload a PDF first.")

        except Exception as e:
            st.error(f"Error generating answer: {e}")
