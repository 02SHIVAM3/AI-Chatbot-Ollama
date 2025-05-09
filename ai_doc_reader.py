import streamlit as st
import faiss
import numpy as np
import pypdf
from langchain_ollama import  OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

llm = OllamaLLM(model="tinyllama")

#load hugging face embeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initialize FAISS Vector Database

index = faiss.IndexFlatL2(384)
vector_store = {}

def extreact_text_from_pdf(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()+"\n"
    return text

# Function to store text in FAISS

def store_in_faiss(text, filename):
    global index, vector_store
    st.write(f"📩 Loading Document '{filename}' in FAISS")

    # split text into chunks
    splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    texts = splitter.split_text(text)
    
    #convert text into embeddings
    
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors,dtype=np.float32)
    
    #store in FAISS
    index.add(vectors)
    vector_store[len(vector_store)] = (filename,texts)
    
    return "Document Stored Successfully !"

# function to retrieve relevant chunks and answer questions

def retrieve_and_answer(query):
    global index,vector_store
    
    #converting querry into embedding
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1,-1)
    
    # search FAISS
    
    D, I = index.search(query_vector,k=2)
    context = ""
    
    for idx in I[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n\n"
    

    
    
    if not context:
        return "NO DATA FOUND"
    
    # ASK AI to generate an answer
    
    return llm.invoke(f"Based on the following document context, answer the question : \n\n")
    
    
#streamlit webUI

st.title("AI DOCUMENT READER")
st.write("Upload a PDF and ask question")

uploaded_file = st.file_uploader("📂 Upload a PDF Document", type=["pdf"])

if uploaded_file:
    text = extreact_text_from_pdf(uploaded_file)
    store_message = store_in_faiss(text,uploaded_file.name)
    st.write(store_message)
    
query = st.text_input("ASK question based on uploaded file : ")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("AI RESPONSE : ")
    st.write(answer)


 