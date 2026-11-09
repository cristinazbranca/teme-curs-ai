import pytest
from pydantic import ValidationError

from tema1 import StudentValidator, citeste_student, valideaza_student


def test_citire_json():
    date = citeste_student()
    assert isinstance(date, dict)
    assert date["nume"] == "Ana Popescu"
    assert date["varsta"] == 20
    assert date["email"] == "ana.popescu@example.com"


def test_validare_student_din_fisier():
    student = valideaza_student(citeste_student())
    assert isinstance(student, StudentValidator)
    assert student.nume == "Ana Popescu"
    assert student.varsta == 20
    assert student.email == "ana.popescu@example.com"


@pytest.mark.parametrize("email", ["ana.popescu", "ana@", "@example.com", "ana popescu@example.com"])
def test_email_invalid(email):
    with pytest.raises(ValidationError):
        valideaza_student({"nume": "Ana Popescu", "varsta": 20, "email": email})


@pytest.mark.parametrize("varsta", [0, -1, -20])
def test_varsta_trebuie_pozitiva(varsta):
    with pytest.raises(ValidationError):
        valideaza_student({"nume": "Ana Popescu", "varsta": varsta, "email": "ana.popescu@example.com"})


def test_varsta_text_este_respinsa():
    with pytest.raises(ValidationError):
        valideaza_student({"nume": "Ana Popescu", "varsta": "20", "email": "ana.popescu@example.com"})


def test_camp_lipsa():
    with pytest.raises(ValidationError):
        valideaza_student({"nume": "Ana Popescu", "varsta": 20})