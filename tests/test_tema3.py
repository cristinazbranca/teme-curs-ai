import os
import uuid

import chromadb
import numpy as np
import pytest
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tema3 import (CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, answer_question, build_context, chunk_text,
                   has_source, index_file, index_folder, parse_input, read_pages, search)

load_dotenv()


class FakeModel:
    """Model de embeddings „de jucărie”, ca testele să ruleze rapid și fără internet.
    Transformă textul într-un vector cu frecvența fiecărei litere."""

    def encode(self, texts, normalize_embeddings=True):
        single = isinstance(texts, str)
        vectors = []
        for text in ([texts] if single else texts):
            v = np.array([text.lower().count(ch) for ch in "abcdefghijklmnopqrstuvwxyz"], dtype=float) + 1.0
            vectors.append(v / np.linalg.norm(v))
        return vectors[0] if single else np.array(vectors)


@pytest.fixture
def collection():
    # o colecție nouă, în memorie, pentru fiecare test
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(name=f"test_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "ana.txt").write_text("Ana are mere. " * 100, encoding="utf-8")
    (tmp_path / "ion.txt").write_text("Ion are pere.", encoding="utf-8")
    return tmp_path


def test_chunk_text_marime_si_suprapunere():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = chunk_text(text)
    assert len(chunks) == 3                       # încep la 0, 680, 1360
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]


def test_parse_input():
    request = parse_input("paragraphs.txt, Despre cine e vorba in acel text?")
    assert request.source == "paragraphs.txt"
    assert request.question == "Despre cine e vorba in acel text?"


def test_parse_input_virgula_in_intrebare():
    request = parse_input("StarMaker.pdf, Cine, cand si unde?")
    assert request.question == "Cine, cand si unde?"


@pytest.mark.parametrize("text", ["fara virgula", "paragraphs.txt,", ", intrebare"])
def test_parse_input_invalid(text):
    with pytest.raises((ValueError, ValidationError)):
        parse_input(text)


def test_read_pages_txt_si_pdf():
    assert read_pages(DATA_DIR / "paragraphs.txt")[0][0] == 1
    pdf_pages = read_pages(DATA_DIR / "invoices.pdf")
    assert len(pdf_pages) >= 1
    assert pdf_pages[0][0] == 1


def test_index_folder_pune_source_in_metadate(collection, folder):
    report = index_folder(collection, FakeModel(), folder)
    assert set(report) == {"ana.txt", "ion.txt"}
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    assert {m["source"] for m in metadatas} == {"ana.txt", "ion.txt"}
    assert all("page" in m for m in metadatas)


def test_has_source(collection, folder):
    index_folder(collection, FakeModel(), folder)
    assert has_source(collection, "ana.txt")
    assert not has_source(collection, "nu_exista.pdf")


def test_search_cauta_doar_in_sursa_data(collection, folder):
    index_folder(collection, FakeModel(), folder)
    results = search(collection, FakeModel(), "ion.txt", "Ce are Ion?")
    assert len(results) == 1
    assert all(r["source"] == "ion.txt" for r in results)
    assert "pere" in results[0]["text"]


def test_reindexarea_nu_dubleaza_chunk_urile(collection, folder):
    model = FakeModel()
    first = index_file(collection, model, folder / "ana.txt")
    index_file(collection, model, folder / "ana.txt")
    assert collection.count() == first


def test_build_context_contine_sursa_si_pagina():
    context = build_context([{"text": "Ion are pere.", "source": "ion.txt", "page": 1, "distance": 0.1}])
    assert "Source: ion.txt, page 1" in context


@pytest.mark.skipif(not os.getenv("OPEN_ROUTER_API_KEY"), reason="lipsește OPEN_ROUTER_API_KEY în .env")
def test_agentul_raspunde_cu_fisierul(collection):
    index_folder(collection, FakeModel(), DATA_DIR)
    results = search(collection, FakeModel(), "paragraphs.txt", "Despre cine e vorba in acel text?")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPEN_ROUTER_API_KEY"))
    raspuns = answer_question(client, "Despre cine e vorba in acel text?", results)
    assert "marcus" in raspuns.lower()
    assert "paragraphs.txt" in  raspuns