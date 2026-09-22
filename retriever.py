from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def main():
	embeddings = OllamaEmbeddings(model="nomic-embed-text")
	vectorstore = Chroma(
		persist_directory="db/chomavectorstore",
		collection_name="documents_cosine",
		embedding_function=embeddings,
	)

	query = "What products does Tesla make?"

	query_vector = embeddings.embed_query(query)
	search = vectorstore._collection.query(
		query_embeddings=[query_vector],
		n_results=3,
		include=["documents", "metadatas", "distances"],
	)

	print(f"Query: {query}\n")
	print("Cosine similarity results:")
	for index, (document, distance) in enumerate(
		zip(search["documents"][0], search["distances"][0]), start=1
	):
		cosine_score = 1 - distance
		print(f"\n{index}. Cosine similarity: {cosine_score:.4f}")
		print(document[:300].replace("\n", " "))


if __name__ == "__main__":
	main()
