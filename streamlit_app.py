import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")
st.set_page_config(page_title="Simple RAG Q&A", layout="centered")
st.title("Document Q&A with RAG")

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.3)
    chunk_size = st.number_input("Chunk size", 500, 2000, 1000)
    st.info("small chunk size and temperature is suggested for a better and a faster output of the document ")

uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if uploaded_file:
    with st.spinner("Processing document..."):
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_file)
        pages = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
        
        embeddings = HuggingFaceEmbeddings()
        st.session_state.vectorstore = Chroma.from_documents(
            texts, 
            embeddings
        )
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
        
        os.remove(temp_file)
        st.success("Document ready for questions!")

question = st.text_input("Ask a question about the document:")

if question and st.session_state.retriever:
    with st.spinner("Thinking..."):
        docs = st.session_state.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=temperature,
            groq_api_key=GROQ_API_KEY
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Answer this question: {question}
            Using only this context: {context}"""
        )
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question, "context": context})
        
        st.subheader("Answer")
        st.write(response)
        
        if st.checkbox("Show source context"):
            st.text_area("Context used:", value=context, height=200)