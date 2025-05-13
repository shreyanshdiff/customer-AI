from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader('./data', glob="**/*.pdf")
documents = loader.load()

text_splitter =  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.vesctorstores import chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_documents(texts, embeddings , persist_directory="./chroma_db")

from langchain.retrievers import BM25Retriever , EnsembleRetriever
from langchain.vectorstores import FAISS
bm25_retriever = BM25Retriever.from_documents(texts)

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5])

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-70b-8192" , temperature = 0.1,
               api_key = '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')
template = """
Answer the question based only on the following context:
{context}
Question: {question}
"""
prompy = ChatPromptTemplate.from_template(template)
rag_chain = prompt | llm | StrOutputParser()

context = "Retrieved documents here..."
question = "What is RAG?"
response = rag_chain.invoke({"context": context, "question": question})
print(response)
