"""Basic stitcher usage — a simple Person extraction.

Run with credentials configured for your chosen LLM provider, e.g.:

    GOOGLE_APPLICATION_CREDENTIALS=... python examples/basic_usage.py

This example uses a small schema with a non-trivial cross-field validator
to demonstrate the JSON-Patch repair loop in action.
"""
from __future__ import annotations

import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, model_validator

from stitcher import AggregatedValidationError, Extractor


class Person(BaseModel):
    name: str = Field(..., description="full name")
    age: int = Field(..., ge=0, le=130)
    email: str
    skills: list[str] = Field(..., description="3-7 skills")

    @model_validator(mode="after")
    def _skills_count(self):
        if not (3 <= len(self.skills) <= 7):
            raise AggregatedValidationError(
                f"skills must contain 3-7 entries, got {len(self.skills)}",
                count=1,
            )
        return self


async def main() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    extractor = Extractor(llm, Person)

    result = await extractor.ainvoke([
        {
            "role": "system",
            "content": "Extract the person described in the user message as a JSON Person object.",
        },
        {
            "role": "user",
            "content": (
                "Alice Carter is a 34-year-old software engineer reachable at "
                "alice@example.com. She is fluent in Python, Rust, SQL, and "
                "system design."
            ),
        },
    ])

    print(f"Value:           {result.value!r}")
    print(f"Attempts:        {result.attempts}")
    print(f"Was re-extracted: {result.was_re_extracted}")


if __name__ == "__main__":
    asyncio.run(main())
