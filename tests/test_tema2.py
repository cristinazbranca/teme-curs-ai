import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

from tema2 import list_files, run_agent, search_keyword

load_dotenv()


def test_list_files_contine_fisierele_proiectului():
    rezultat = list_files()
    assert "tema1.py" in rezultat
    assert "tema2.py" in rezultat
    assert "pyproject.toml" in rezultat
    assert ".venv" not in rezultat   # pachetele instalate nu se listează


def test_search_keyword_gaseste_pytest():
    rezultat = search_keyword({"keyword": "pytest"})
    # aceste fișiere conțin sigur cuvântul "pytest"
    assert "pyproject.toml" in rezultat
    assert "tests/test_tema1.py" in rezultat


def test_search_keyword_nu_tine_cont_de_majuscule():
    assert "pyproject.toml" in search_keyword({"keyword": "PyTest"})


def test_search_keyword_cuvant_inexistent():
    # construim cuvântul din bucăți, altfel l-ar găsi chiar în acest fișier de test
    cuvant = "q" * 3 + "_lipsa_" + "9" * 3
    assert "No files contain" in search_keyword({"keyword": cuvant})


@pytest.mark.skipif(not os.getenv("OPEN_ROUTER_API_KEY"), reason="lipsește OPEN_ROUTER_API_KEY în .env")
def test_agentul_returneaza_fisierele_cu_pytest():
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPEN_ROUTER_API_KEY"))
    raspuns = run_agent(client, "pytest")
    assert "pyproject.toml" in raspuns