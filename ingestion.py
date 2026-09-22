import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split documents into chunks"""
    print(f"\nSplitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print(f"  Content length: {len(chunk.page_content)} characters")
        print(f"  Content preview: {chunk.page_content[:100]}...")
        print(f"  metadata: {chunk.metadata}")

    return chunks

def create_vectorstore(chunks, persist_directory="db/chomavectorstore"):
    """Create a vector store from document chunks"""
    print("\nCreating vector store...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory,
        collection_name="documents_cosine",
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created with {vectorstore._collection.count()} vectors.")
    
    return vectorstore


if __name__ == "__main__":
    documents=load_documents(docs_path="docs")
    chunks=split_documents(documents)
    vectorstore = create_vectorstore(chunks)
