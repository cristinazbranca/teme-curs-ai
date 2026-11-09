
import json
from pathlib import Path

from pydantic import BaseModel, EmailStr, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent


class StudentValidator(BaseModel):
    nume: str = Field(min_length=1, description="Numele studentului.")
    varsta: int = Field(gt=0, description="Vârsta studentului, număr întreg pozitiv.")
    email: EmailStr = Field(description="Adresa de email a studentului.")


def citeste_student(nume_fisier: str = "student.json") -> dict:
    with open(PROJECT_ROOT / nume_fisier, "r", encoding="utf-8") as f:
        return json.load(f)


def valideaza_student(date: dict) -> StudentValidator:
    return StudentValidator.model_validate(date, strict=True)


if __name__ == "__main__":
    date = citeste_student()
    try:
        student = valideaza_student(date)
        print("Student valid:", student)
    except ValidationError as e:
        print("Date invalide:")
        print(e)