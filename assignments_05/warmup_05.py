"""Week 5 practice: Chat Completions, prompts, and local models."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"


def ask(messages, **options):
    """Send a chat completion request and return the response object."""
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        **options,
    )


# --- Completions API ---
# API Q1
response = ask([
    {"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}
])
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
for temperature in (0, 0.7, 1.5):
    response = ask([{"role": "user", "content": prompt}], temperature=temperature)
    print(f"Temperature {temperature}: {response.choices[0].message.content}")
# A low temperature tends to give more predictable wording, while a higher one
# usually gives more varied ideas. I'd use 0 when I need a consistent result.

# API Q3
response = ask(
    [{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0,
)
for number, choice in enumerate(response.choices, start=1):
    print(f"Completion {number}: {choice.message.content}")

# API Q4
response = ask(
    [{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15,
)
print("Limited response:", response.choices[0].message.content)
# The answer gets cut short at the token limit. In an application this can cap
# response length and cost, though the limit needs to leave room for a full answer.


# --- System Messages and Personas ---
# System Q1
question = "I don't understand what a list comprehension is."
for persona in (
    "You are a patient, encouraging Python tutor. Explain simply and end with encouragement.",
    "You are a terse code reviewer. Give a compact, practical explanation without small talk.",
):
    response = ask([
        {"role": "system", "content": persona},
        {"role": "user", "content": question},
    ])
    print(f"\nPersona: {persona}\nResponse: {response.choices[0].message.content}")
# The first answer should feel reassuring; the second should be more direct.

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = ask(messages)
print("Conversation memory response:", response.choices[0].message.content)
# The model can answer because the earlier messages were sent again in this call.


# --- Prompt Engineering ---
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]


def classify_reviews(instructions, examples=()):
    """Classify the reviews using the requested amount of examples."""
    example_text = "\n".join(examples)
    review_text = "\n".join(f"Review {i}: {review}" for i, review in enumerate(reviews, 1))
    prompt = f"""{instructions}
Classify every review as positive, negative, or mixed. Keep the answer concise and
label each result by review number.

Examples:
{example_text}

Reviews:
{review_text}"""
    response = ask([{"role": "user", "content": prompt}])
    print(response.choices[0].message.content)


# Prompt Q1 — Zero-Shot
print("\nZero-shot results:")
classify_reviews("Classify the sentiment of each review.")

# Prompt Q2 — One-Shot
print("\nOne-shot results:")
classify_reviews(
    "Classify the sentiment of each review.",
    ['Review: "Fast shipping but the item arrived damaged."\nSentiment: mixed'],
)
# One example can establish a compact format and show how to handle a mixed review.

# Prompt Q3 — Few-Shot
print("\nFew-shot results:")
classify_reviews(
    "Classify the sentiment of each review.",
    [
        'Review: "The team was kind and the service was excellent."\nSentiment: positive',
        'Review: "The app loses my work and nobody answered my ticket."\nSentiment: negative',
        'Review: "The price is fair, but setup was frustrating."\nSentiment: mixed',
    ],
)
# Zero-shot is quick for an easy task; one example shows a desired answer style;
# few-shot is useful when categories are subtle or consistency matters more.

# Prompt Q4 — Chain of Thought
response = ask([{
    "role": "user",
    "content": (
        "Solve this carefully and give a brief calculation summary followed by a clearly labeled final answer. "
        "A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later "
        "takes a new job that pays $7,500 more per year than her post-raise salary. "
        "What is her final annual salary?"
    ),
}])
print("Salary calculation:", response.choices[0].message.content)
# Breaking the salary change into two calculations helps reduce arithmetic slips.

# Prompt Q5 — Structured Output
review = (
    "I've been using this tool for three months. It handles large datasets well, "
    "but the UI is clunky and the export options are limited."
)
response = ask([{
    "role": "user",
    "content": f'''Analyze this review and respond only with valid JSON, no markdown.
Use keys sentiment, confidence (a number from 0 to 1), and reason (one sentence).
Review: """{review}"""''',
}])
raw = response.choices[0].message.content or ""
print("Raw JSON response:", raw)
try:
    parsed = json.loads(raw)
except json.JSONDecodeError as error:
    print("Could not parse JSON; raw response is shown above.", error)
else:
    for key in ("sentiment", "confidence", "reason"):
        print(f"{key.title()}:", parsed.get(key))

# Prompt Q6 — Delimiters
user_text = (
    "First boil a pot of water. Once boiling, add a handful of salt and the pasta. "
    "Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
)
for text in (
    user_text,
    "A quiet rain fell over the garden while the evening light faded behind the hills.",
):
    prompt = f'''You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{text}```'''
    response = ask([{"role": "user", "content": prompt}])
    print("Delimited text result:", response.choices[0].message.content)
# Delimiters make it clearer which text is data to process and which text is the task.


# --- Local Models with Ollama ---
# Ollama Q1
# Run `ollama run qwen3:0.6b "Explain what a large language model is in two sentences."`
# and paste the output here after trying it locally. I haven't recorded an output
# in this file, so this comment doesn't claim a result I didn't observe.
# I expect the local model to work without sending the prompt to an API, though
# its speed and answer quality depend on the computer and downloaded model.
response = ask([{
    "role": "user",
    "content": "Explain what a large language model is in two sentences.",
}])
print("OpenAI response:", response.choices[0].message.content)
