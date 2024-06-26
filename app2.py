import chromadb.utils
import chromadb.utils.embedding_functions
import chromadb.utils.fastapi
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import openai
import torch
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import openai

kChromaURL = '#.#.#.#' # removed for security
kChromaPort = 0 # removed for security
kCollectionName = 'skills'
kMaxTokens = 1000

kBaseURL = '192.168.178.34'

chromaClient = chromadb.HttpClient(host=kChromaURL, port=kChromaPort)

# Step 1: Data Ingestion
def ingest_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

# Step 2: Embedding Generation
def generate_embeddings(text_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    embeddings = model.encode(text_data, show_progress_bar=True)
    return embeddings

# Step 3: Database Storage
def store_embeddings(embeddings, metadata):
    try:
       collection = chromaClient.get_collection(kCollectionName)
    except:
        collection = chromaClient.create_collection(kCollectionName) 
    
    for embedding, meta in zip(embeddings, metadata):
        # collection.add(chromadb.utils.embedding_to_dict(embedding), meta)
        collection.add(embeddings=embedding, metadatas=metadata)

# Step 4: Retrieval Logic
def retrieve_chunks(query, top_k=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    query_embedding = model.encode([query])

    collection = chromaClient.get_collection(kCollectionName)
    
    results = collection.query(query_embedding, top_k)
    return results

def generate_response(query, chunks):
    combined_text = query + " " + " ".join([chunk['job_description'] for chunk in chunks])
    openai.AzureOpenAI.openai_api_base = kBaseURL
    response = openai.Chat.create(
        prompt = combined_text,
        max_tokens = kMaxTokens,
    )
    
    response = openai.Completion.create(
        prompt=combined_text
        max_tokens=150
    )
    return response.choices[0].text

def main(file_path, query):
    job_descriptions = ingest_data(file_path)
    embeddings = generate_embeddings(job_descriptions)
    store_embeddings(embeddings, [{"job_description": jd} for jd in job_descriptions])
    relevant_chunks = retrieve_chunks(query)
    response = generate_response(query, relevant_chunks)
    return response

if __name__ == "__main__":
    file_path = 'linkedin_skills/small.txt'
    query = "data engineer"
    response = main(file_path, query)
    print(response)