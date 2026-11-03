"""Week 5 API, persona, and prompt-engineering warmups."""

import json
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / ".env")
client = OpenAI()
MODEL = "gpt-4o-mini"


def ask(prompt: str, *, system: str | None = None, **options):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return client.chat.completions.create(model=MODEL, messages=messages, **options)


def print_review_labels(response, label: str) -> None:
    """Print one completion with a label."""
    print(f"{label}: {response.choices[0].message.content.strip()}")


def run_warmups() -> None:
    # --- Completions API ---
    # API Q1
    response = ask("What is one thing that makes Python a good language for beginners?")
    print("Response:", response.choices[0].message.content.strip())
    print("Model:", response.model)
    print("Total tokens:", response.usage.total_tokens)

    # API Q2
    prompt = "Suggest a creative name for a data engineering consultancy."
    for temperature in [0, 0.7, 1.5]:
        response = ask(prompt, temperature=temperature)
        print_review_labels(response, f"Temperature {temperature}")
    # Higher temperatures usually produce more varied wording; use 0 when a
    # consistent, reproducible style matters.

    # API Q3
    response = ask(
        "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        n=3,
        temperature=1.0,
    )
    for index, choice in enumerate(response.choices, start=1):
        print(f"Completion {index}:", choice.message.content.strip())

    # API Q4
    response = ask("Explain how neural networks work.", max_tokens=15)
    print("Max-token response:", response.choices[0].message.content.strip())
    # The response stops at the token limit, which helps control response
    # length, latency, and API cost in an application.

    # --- System Messages and Personas ---
    # System Q1
    question = "I don't understand what a list comprehension is."
    patient_tutor = (
        "You are a patient, encouraging Python tutor. Explain simply and end "
        "with a word of encouragement."
    )
    terse_expert = (
        "You are a concise Python reference manual. Explain with a precise "
        "definition and one compact code example. Avoid encouragement or small talk."
    )
    response = ask(question, system=patient_tutor)
    print("Patient tutor:", response.choices[0].message.content.strip())
    response = ask(question, system=terse_expert)
    print("Concise reference:", response.choices[0].message.content.strip())
    # The tutor is supportive and plain-spoken; the reference persona is terse
    # and example-focused, so the same question gets a different presentation.

    # System Q2
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My name is Jordan and I'm learning Python."},
        {
            "role": "assistant",
            "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
        },
        {"role": "user", "content": "Can you remind me what my name is?"},
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    print("Conversation memory:", response.choices[0].message.content.strip())
    # The model receives Jordan's name in the supplied history on this call;
    # stateless means it does not retain the prior call automatically.

    # --- Prompt Engineering ---
    reviews = [
        "The onboarding process was smooth and the team was welcoming.",
        "The software crashes constantly and support never responds.",
        "Great price, but the documentation is nearly impossible to follow.",
    ]

    # Prompt Q1 — Zero-Shot
    review_block = "\n".join(f"{index}. {review}" for index, review in enumerate(reviews, start=1))
    zero_shot_prompt = (
        "Classify each review as positive, negative, or mixed. Return one label "
        "per review, preserving the numbering.\n\n" + review_block
    )
    response = ask(zero_shot_prompt)
    print("Zero-shot results:\n" + response.choices[0].message.content.strip())

    # Prompt Q2 — One-Shot
    one_shot_prompt = (
        'Classify each review as positive, negative, or mixed. Return one label per review, preserving the numbering.\n'
        'Example:\nReview: "Fast shipping but the item arrived damaged."\nSentiment: mixed\n\n'
        + review_block
    )
    response = ask(one_shot_prompt)
    print("One-shot results:\n" + response.choices[0].message.content.strip())
    # One example can make the response format and the meaning of "mixed"
    # more consistent with the format the user expects.

    # Prompt Q3 — Few-Shot
    few_shot_prompt = (
        "Classify each review as positive, negative, or mixed. Return one label per review, preserving the numbering.\n"
        'Example: "The staff were kind and solved my problem." → positive\n'
        'Example: "The app loses my work every time I open it." → negative\n'
        'Example: "The price is fair, but setup took all afternoon." → mixed\n\n'
        + review_block
    )
    response = ask(few_shot_prompt)
    print("Few-shot results:\n" + response.choices[0].message.content.strip())
    # Zero-shot is quick for simple common tasks; one-shot demonstrates a
    # format; few-shot is useful when examples clarify edge cases and labels.

    # Prompt Q4 — Chain of Thought
    math_prompt = (
        "Solve this salary problem. Show a concise calculation breakdown, then "
        "label the final answer clearly. A data engineer earns $85,000 per year. "
        "She gets a 12% raise, then 6 months later takes a new job that pays "
        "$7,500 more per year than her post-raise salary. What is her final annual salary?"
    )
    response = ask(math_prompt)
    print("Salary solution:\n" + response.choices[0].message.content.strip())
    # Asking for intermediate calculations makes the arithmetic steps explicit,
    # which can reveal mistakes and makes the result easier to check.

    # Prompt Q5 — Structured Output
    review = (
        "I've been using this tool for three months. It handles large datasets well, "
        "but the UI is clunky and the export options are limited."
    )
    json_prompt = (
        "Analyze this review and return only valid JSON with keys sentiment, "
        "confidence (a float from 0 to 1), and reason (one sentence).\nReview: "
        + review
    )
    response = ask(json_prompt, response_format={"type": "json_object"})
    raw_json = response.choices[0].message.content
    print("Raw JSON response:", raw_json)
    try:
        review_analysis = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as error:
        print("Could not parse the response as JSON:", error)
        print("Raw response for debugging:", raw_json)
    else:
        for field in ("sentiment", "confidence", "reason"):
            print(f"{field.title()}:", review_analysis[field])

    # Prompt Q6 — Delimiters
    instruction_text = (
        "First boil a pot of water. Once boiling, add a handful of salt and the "
        "pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
    )
    delimiter_prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{instruction_text}```"""
    response = ask(delimiter_prompt)
    print("Delimited instructions:\n" + response.choices[0].message.content.strip())

    prose_text = "The city library opened a new reading room beside the river last spring."
    prose_prompt = f"""You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{prose_text}```"""
    response = ask(prose_prompt)
    print("Delimited prose:", response.choices[0].message.content.strip())
    # Delimiters separate untrusted text from the instruction, reducing the
    # chance that quoted content is mistaken for directions to the model.

    # --- Local Models with Ollama ---
    # Ollama Q1
    ollama_prompt = "Explain what a large language model is in two sentences."
    if shutil.which("ollama"):
        ollama_result = subprocess.run(
            ["ollama", "run", "qwen3:0.6b", ollama_prompt],
            check=True,
            capture_output=True,
            text=True,
        )
        ollama_output = ollama_result.stdout.strip()
        print("Ollama response:", ollama_output)
    else:
        ollama_output = "Ollama was not installed in this environment; no local output was captured."
        print(ollama_output)

    response = ask(ollama_prompt)
    print("OpenAI response:", response.choices[0].message.content.strip())
    # Ollama output (qwen3:0.6b):
    # "A large language model is an artificial intelligence system trained on
    # vast amounts of text to understand and generate human-like responses. It
    # excels in tasks like writing, answering questions, and understanding
    # complex topics, making it powerful for various applications."
    # This answer is broad and use-focused. Running locally can keep prompts
    # on-device and work offline, but may be less capable than a hosted model.


if __name__ == "__main__":
    run_warmups()
