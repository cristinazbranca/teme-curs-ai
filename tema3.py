import os
from pathlib import Path

import chromadb
import pymupdf4llm
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_sources"
DB_DIR = PROJECT_ROOT / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "data_sources_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
N_RESULTS = 3
MAX_DISTANCE = 0.5

SYSTEM_PROMPT = """Answer only from the supplied context. If the context does not answer the question, say so.
For every piece of information, say the file name and the page number it comes from,
using the 'Source' line under each context fragment. Answer in the same language as the question."""


class UserQuestion(BaseModel):
    source: str = Field(min_length=1, description="Numele fișierului din data_sources.")
    question: str = Field(min_length=1, description="Întrebarea utilizatorului.")


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start = start + CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def read_pages(file_path: Path) -> list[tuple[int, str]]:
    if file_path.suffix.lower() == ".pdf":
        pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        return [(page["metadata"]["page_number"], page["text"]) for page in pages]
    return [(1, file_path.read_text(encoding="utf-8"))]


def index_file(collection, model, file_path: Path) -> int:
    ids, documents, metadatas = [], [], []
    for page_number, page_text in read_pages(file_path):
        for chunk_number, chunk in enumerate(chunk_text(page_text)):
            ids.append(f"{file_path.name}:p{page_number}:c{chunk_number}")
            documents.append(chunk)
            metadatas.append({"source": file_path.name, "page": page_number, "chunk": chunk_number})

    collection.delete(where={"source": file_path.name})
    if documents:
        collection.upsert(
            ids=ids,
            embeddings=model.encode(documents, normalize_embeddings=True).tolist(),
            documents=documents,
            metadatas=metadatas,
        )
    return len(documents)


def index_folder(collection, model, folder: Path = DATA_DIR) -> dict[str, int]:
    report = {}
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and not file_path.name.startswith("."):
            try:
                report[file_path.name] = index_file(collection, model, file_path)
            except (UnicodeDecodeError, ValueError) as error:
                print(f"skip> {file_path.name}: {error}")
    return report


def parse_input(text: str) -> UserQuestion:
    if "," not in text:
        raise ValueError("Scrie așa: <nume fișier>, <întrebare>")
    source, question = text.split(",", 1)
    return UserQuestion(source=source.strip(), question=question.strip())


def has_source(collection, source: str) -> bool:
    result = collection.get(where={"source": source}, limit=1)
    return len(result["ids"]) > 0


def search(collection, model, source: str, question: str, n_results: int = N_RESULTS) -> list[dict]:
    query_embedding = model.encode(question, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"source": source},
        include=["documents", "metadatas", "distances"],
    )
    found = []
    for i in range(len(results["documents"][0])):
        metadata = results["metadatas"][0][i]
        found.append({
            "text": results["documents"][0][i],
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": results["distances"][0][i],
        })
    return found


def filter_by_distance(results: list[dict], max_distance: float = MAX_DISTANCE) -> list[dict]:
    return [r for r in results if r["distance"] <= max_distance]


def build_context(results: list[dict]) -> str:
    pieces = [f"{r['text']}\nSource: {r['source']}, page {r['page']}" for r in results]
    return "\n\n".join(pieces)


def answer_question(client: OpenAI, question: str, results: list[dict]) -> str:
    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{build_context(results)}"},
        ],
    )
    return response.choices[0].message.content or ""


def main():
    load_dotenv()
    from sentence_transformers import SentenceTransformer

    print("start> încarc modelul de embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPEN_ROUTER_API_KEY"))

    report = index_folder(collection, model)
    for name, count in report.items():
        print(f"index> {name}: {count} chunk-uri")

    print('\nScrie: <nume fișier>, <întrebare>   (sau "quit" ca să ieși)')
    while True:
        text = input("you> ").strip()
        if text == "quit":
            break
        try:
            request = parse_input(text)
        except (ValueError, ValidationError) as error:
            print(f"error> {error}")
            continue

        if not has_source(collection, request.source):
            print(f"agent> Nu am date despre '{request.source}'. Fișiere disponibile: {', '.join(report)}")
            continue

        results = search(collection, model, request.source, request.question)
        relevant = filter_by_distance(results)
        for r in results:
            status = "păstrat" if r in relevant else f"ignorat, peste {MAX_DISTANCE}"
            print(f"found> {r['source']}, pagina {r['page']}, distanța {r['distance']:.3f} ({status})")

        if not relevant:
            print("agent> Nu s-a găsit niciun rezultat relevant: toate rezultatele sunt la o distanță prea mare.")
            continue
        print("agent>", answer_question(client, request.question, relevant))


if __name__ == "__main__":
    main()