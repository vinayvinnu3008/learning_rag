import sys

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
VECTORSTORE_PATH = "db/chomavectorstore"
COLLECTION_NAME = "documents_cosine"


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Ask a question about the documents: ").strip()
    if not query:
        print("Please enter a question.")
        return

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    documents = vectorstore.similarity_search(query, k=4)
    if not documents:
        print("No documents were found. Run ingestion.py first.")
        return

    context = "\n\n---\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in documents
    )

    prompt = f"""You answer questions using only the supplied context.
If the answer is not in the context, say: I don't know based on the documents.
Do not invent facts.

Context:
{context}

Question: {query}

Answer:"""

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)
    print("\nSources:")
    for document in documents:
        print(f"- {document.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
