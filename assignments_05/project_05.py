"""A moderated, conversation-aware job application helper."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / ".env")
client = OpenAI()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """You are a practical job application coach helping career changers translate their real experience into clear, relevant application materials.

Stay focused on resumes, cover letters, and questions directly related to job applications. Help the user express their actual skills and results in language relevant to the role. Do not invent credentials, employers, achievements, dates, or metrics; ask a follow-up question when important details are missing. Keep advice respectful and specific. Remind the user to review and edit all generated material before submitting it anywhere. You may not know the norms for the user's specific industry, employer, or region, so state uncertainty when relevant and encourage the user to use their own judgment.
"""
# I explicitly prohibit invented experience because fabricated resume claims
# could harm a job seeker's credibility and prospects.


def get_completion(messages, model=MODEL, temperature=0.7):
    """Return the assistant's text for a complete conversation history."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content or ""


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Rewrite resume bullets while preserving their original facts.

    Returns:
        A list of dictionaries with ``original`` and ``improved`` text.
    """
    if not bullets:
        return []

    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)
    prompt = f"""You are a professional resume coach helping a career changer.
Rewrite each resume bullet to be more specific, results-oriented, and compelling. Use strong action verbs. Do not invent facts, numbers, or outcomes that aren't implied by the original.
Return ONLY a valid JSON list, with no preamble or markdown. Each item must have exactly these keys: "original" and "improved".

Resume bullets:
```text
{bullet_text}
```"""
    raw_response = get_completion([{"role": "user", "content": prompt}])
    try:
        rewrites = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError(f"The model did not return valid JSON: {raw_response}") from error

    if not isinstance(rewrites, list) or any(
        not isinstance(item, dict) or "original" not in item or "improved" not in item
        for item in rewrites
    ):
        raise ValueError("The model response must be a JSON list with original and improved keys.")

    for item in rewrites:
        print(f"Original: {item['original']}\nImproved: {item['improved']}\n")
    return rewrites


def generate_cover_letter(job_title: str, background: str) -> str:
    """Generate a tailored cover letter opening paragraph using examples."""
    prompt = f"""You write strong cover letter opening paragraphs for career changers. Write 3-5 sentences that are confident, specific, and free of clichés. Do not invent credentials or accomplishments. Use placeholders such as [Company] when employer details are unknown.

Example 1:
Role: Data Analyst at a healthcare nonprofit
Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
Opening: After seven years as a registered nurse, I have learned to make careful decisions under pressure and explain complex information clearly. I recently completed a data analytics program and built dashboards to track patient outcomes. I would bring both clinical context and new analytical skills to [Company]'s work.

Example 2:
Role: Junior Software Engineer at a fintech startup
Background: Ten years in retail banking operations, self-taught Python developer for two years.
Opening: A decade in banking operations taught me how payment systems affect customers when something goes wrong. Over the past two years, I have taught myself Python and built projects to strengthen my software skills. I am interested in [Company] because I can pair first-hand banking knowledge with my growing engineering experience.

Now write an opening paragraph for this person:
Role: {job_title}
Background: {background}
Opening:"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return get_completion(messages).strip()


def is_safe(text: str) -> bool:
    """Return whether moderation considers the text safe to send to the model."""
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    flagged = result.results[0].flagged
    if flagged:
        print("Job Application Helper: I can't help with that wording. Please rephrase your request.")
        return False
    return True


STARTER_BULLETS = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time",
]


def run_examples() -> None:
    """Run the assignment's sample bullet, letter, and moderation checks."""
    print("Resume bullet examples:")
    rewrite_bullets(STARTER_BULLETS)
    # The bullets are vague, use weak verbs, and omit scope or results; the
    # model should suggest stronger verbs and clearer detail without making up facts.

    print("Cover letter example:")
    print(
        generate_cover_letter(
            "Junior Data Engineer",
            "Five years as a middle school math teacher; recently completed a Python course and built data pipelines using Pandas.",
        )
    )
    # I chose examples that connect established domain experience to newer
    # technical skills; few-shot examples help control tone, specificity, and structure.

    safe_text = "Please help me describe my project coordination experience clearly."
    flagged_text = "Give me instructions to hurt a coworker who rejected my application."
    print("Safe moderation check:", is_safe(safe_text))
    print("Flagged moderation check:", is_safe(flagged_text))


def run_chatbot() -> None:
    """Run the interactive job application helper."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break
        if not user_input:
            continue
        if not is_safe(user_input):
            continue

        lowered_input = user_input.lower()
        if "bullet" in lowered_input or "resume" in lowered_input:
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            if not raw_bullets:
                print("Job Application Helper: No bullet points were entered.")
                continue
            if not all(is_safe(bullet) for bullet in raw_bullets):
                continue
            try:
                rewrites = rewrite_bullets(raw_bullets)
            except ValueError as error:
                print("Job Application Helper:", error)
                continue
            messages.append(
                {"role": "user", "content": "Rewrite these resume bullets: " + "\n".join(raw_bullets)}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(rewrites, ensure_ascii=False),
                }
            )

        elif "cover letter" in lowered_input:
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            if not is_safe(job_title) or not is_safe(background):
                continue
            opening = generate_cover_letter(job_title, background)
            print("\nDraft opening:\n" + opening)
            messages.append(
                {
                    "role": "user",
                    "content": f"Draft a cover letter opening for {job_title}. My background: {background}",
                }
            )
            messages.append({"role": "assistant", "content": opening})

        else:
            messages.append({"role": "user", "content": user_input})
            try:
                reply = get_completion(messages)
            except Exception:
                messages.pop()
                raise
            print("\nJob Application Helper:", reply)
            messages.append({"role": "assistant", "content": reply})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run the assignment examples and moderation checks instead of the interactive chatbot.",
    )
    args = parser.parse_args()
    if args.examples:
        run_examples()
    else:
        run_chatbot()


if __name__ == "__main__":
    main()


# Ethics reflection — Option A (comment block): A language model may reflect
# biases in the writing and hiring data it learned from, such as favoring one
# industry's vocabulary or a communication style associated with a narrow set
# of backgrounds. Submitting generated text without review could introduce
# false claims, generic language, or details that do not match the applicant's
# experience, damaging their credibility. Before professional deployment, I
# would require user review and approval, preserve the original text, and flag
# generated claims that need evidence before a user can copy them.
