import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent
KEYWORD = "pytest"
MAX_AGENT_STEPS = 6

IGNORED_FOLDERS = {".venv", ".git", "__pycache__", ".pytest_cache", ".idea"}

SYSTEM_PROMPT = """You are a helpful assistant that searches inside a code project.
Always follow these steps in order:
1. First call the list_files tool to see all files in the project.
2. Then call the search_keyword tool with the keyword the user asks for.
3. Finally answer with the list of file paths where the keyword was found, one per line.
Only report files returned by the tools. Do not invent file names."""


class SearchKeywordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1, description="The keyword to search for, case-insensitive.")


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def project_files() -> list[Path]:
    files = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in IGNORED_FOLDERS for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def list_files() -> str:
    lines = [path.relative_to(PROJECT_ROOT).as_posix() for path in project_files()]
    return "\n".join(lines)


def search_keyword(args: dict) -> str:
    try:
        request = SearchKeywordRequest.model_validate(args)
    except ValidationError as error:
        return f"error: {error}"

    keyword = request.keyword.lower()
    found = []
    for path in project_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if keyword in text.lower():
            found.append(path.relative_to(PROJECT_ROOT).as_posix())

    if not found:
        return f"No files contain '{request.keyword}'."
    return json.dumps({"keyword": request.keyword, "files": found})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists all files in the current project (like `dir /s`). Call this first.",
            "parameters": EmptyRequest.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Searches a keyword (case-insensitive) in the content of all project files "
                           "(like `findstr /s /i`). Returns the files that contain it.",
            "parameters": SearchKeywordRequest.model_json_schema(),
        },
    },
]


def call_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return list_files()
    if name == "search_keyword":
        return search_keyword(args)
    return f"error: unknown tool {name}"


def complete(client: OpenAI, messages: list[dict]):
    return client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    ).choices[0].message


def run_agent(client: OpenAI, keyword: str = KEYWORD) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Find all files in this project that contain the keyword '{keyword}'."},
    ]

    for step in range(MAX_AGENT_STEPS):
        response = complete(client, messages)
        messages.append(response.model_dump(exclude_none=True))

        if not response.tool_calls:
            return response.content or ""

        for tool_call in response.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"tool> {name}({args})")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": call_tool(name, args),
            })

    return "error: agent stopped, step limit reached."


def main():
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )
    answer = run_agent(client)
    print(f"\nagent> Fișierele care conțin '{KEYWORD}':\n{answer}")


if __name__ == "__main__":
    main()