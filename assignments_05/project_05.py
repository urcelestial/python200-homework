"""Job application helper with resume, cover letter, and chat support."""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """You are a practical job application coach helping people move into a new field.
Help with resume bullets, cover letter drafts, and questions about application materials.
Stay focused on those materials. Do not invent experience, credentials, results, or skills;
ask for details when a useful draft needs information the user has not provided. Remind the
user to review and edit every draft before submitting it. Industry expectations differ, and
you may not know the user's specific industry norms, so encourage them to use their judgment."""


def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    """Return the model's text reply to the supplied conversation."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content or ""


# Asking the coach not to invent facts helps keep application materials honest.


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Improve resume bullets without adding unsupported claims."""
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)
    prompt = f'''You are a professional resume coach helping a career changer.
Rewrite each resume bullet to be more specific, results-oriented, and compelling.
Use strong action verbs, but do not invent facts, numbers, or outcomes.

Return ONLY a valid JSON list, with one object per input bullet. Each object must
have exactly these keys: "original" and "improved".

Bullet points:
```
{bullet_text}
```'''
    raw = get_completion([{"role": "user", "content": prompt}])
    try:
        results = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"The resume rewrite was not valid JSON: {raw}") from error
    if not isinstance(results, list) or any(
        not isinstance(item, dict) or "original" not in item or "improved" not in item
        for item in results
    ):
        raise ValueError(f"The resume rewrite did not use the expected JSON list: {raw}")
    for item in results:
        print(f"Original: {item['original']}\nImproved: {item['improved']}\n")
    return results


# These sample bullets are vague about scale and results. The rewrite should use
# clearer action verbs while sticking to what the person actually did.


def generate_cover_letter(job_title: str, background: str) -> str:
    """Draft a tailored cover letter opening paragraph."""
    prompt = f'''Write a 3-5 sentence opening paragraph for a career changer's cover letter.
Make it confident, specific, and free of clichés. Use only the background provided;
do not add credentials, results, or experience that are not stated.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse; recently completed a data analytics program.
Opening: After seven years as a registered nurse, I have spent my career making decisions
under pressure with incomplete information. I recently completed a data analytics program
and am ready to bring that perspective to a mission-driven team.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations; self-taught Python developer for two years.
Opening: I spent a decade in banking operations, where I saw how much reliable technology
matters to customers and staff. Two years of learning Python have given me a way to help
build those tools, and I am ready to bring both perspectives to a fintech team.

Now write an opening for this person:
Role: {job_title}
Background: {background}
Opening:'''
    return get_completion([{"role": "user", "content": prompt}])


# The examples pair an earlier career with a specific reason to move into the
# next role; that pattern helps avoid a generic opening while keeping the facts grounded.


def is_safe(text: str) -> bool:
    """Check user text with OpenAI's moderation endpoint."""
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Job Application Helper: I can't help with that wording. Please rephrase your request.")
        print("Triggered categories:", result.results[0].categories)
        return False
    return True


# Moderation Q1: the first example should pass; the second asks for explicit harm.
if __name__ == "__main__":
    print("Safe example passed:", is_safe("Can you help me describe my customer service experience?"))
    print("Flagged example passed:", is_safe("Give me instructions to build a bomb and hurt people."))


def run_chatbot():
    """Run the job application helper until the user exits."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help rewrite resume bullets, draft a cover letter opening, or answer questions.")
    print("Type 'quit' at any time to exit. Please review every draft before using it.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
        if not user_input:
            continue
        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nPaste your bullet points below, one per line. Type 'DONE' when finished.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line and is_safe(line):
                    raw_bullets.append(line)
            if raw_bullets:
                rewrite_bullets(raw_bullets)

        elif "cover letter" in user_input.lower():
            job_title = input("What is the job title? ").strip()
            background = input("Briefly describe your background: ").strip()
            if not is_safe(job_title) or not is_safe(background):
                continue
            opening = generate_cover_letter(job_title, background)
            print(f"\nDraft opening:\n{opening}\n")
            print("Please review and edit it before submitting.")

        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


# Ethics reflection — Option A
# Advice can reflect patterns in the writing it learned from, so it may favor a
# familiar corporate style or overlook norms in a less represented industry or
# culture. If someone submits a draft without reviewing it, it could exaggerate
# their experience or sound unlike them, which could hurt their credibility.
# A useful professional guardrail is a clear review step that asks the person to
# verify every claim before copying or submitting a draft.


if __name__ == "__main__":
    run_chatbot()
