"""Week 6 warmups: RAG concepts, keyword search, and LlamaIndex."""

from dotenv import load_dotenv
import os
import string
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---
# Concepts Q1
# A: RAG. Retrieve current policy passages from the PDF library so the answers
# can reflect quarterly updates and point back to the relevant material.
# B: Fine-tuning. The 3,000 examples can teach the model a distinctive writing
# style that is not well represented in its general training.
# C: Prompt engineering. Include the short report in the prompt and ask a
# focused question; building a reusable retrieval system would be unnecessary.

# Concepts Q2
# A confident wrong answer is more harmful because its certainty discourages
# people from checking it. For example, a made-up medication dose stated as a
# fact could lead someone to take an unsafe amount. Tone matters because people
# often use confidence as a shortcut for judging whether information is reliable.

# Concepts Q3
# Original unordered list:
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]
# Correct order and description:
# 1. Extract text from source documents — make document contents searchable.
# 2. Split text into chunks — create manageable passages to retrieve.
# 3. Convert text chunks into embeddings — represent each passage as a vector.
# 4. Receive the user's query — get the information need to answer.
# 5. Embed the user's query — represent the question in the same vector space.
# 6. Retrieve the most relevant chunks — find passages closest to the query.
# 7. Inject retrieved chunks into the prompt — provide evidence to the LLM.
# 8. Generate a response from the LLM — answer using the question and evidence.


# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# Keyword Question 1
query = "What are your hours on weekends?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# hours.txt is selected because it shares the query terms "hours" and "weekends".

# Keyword Question 2
query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# The menu is the likely selection because it mentions drinks, but keyword search
# does not understand that "without caffeine" means decaf; semantic retrieval
# can match related meanings even when the exact words differ.

# Keyword Question 3
# Prediction: loyalty.txt, because rewards is closely related to loyalty and sign-up.
query = "How do I sign up for rewards?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# Prediction should be checked against the overlap output. Keyword matching may
# miss the relationship between "rewards" and "loyalty" if no exact terms match.


# --- Semantic RAG Concepts ---
# Semantic Question 1
# A vector embedding is a list of numbers that captures useful aspects of a
# text's meaning. The chunk with similarity 0.85 is more relevant than the one
# scored 0.30; the higher score means the query and chunk point in more similar
# directions in the embedding space. Semantic search can match paraphrases and
# related concepts because it compares meaning representations, not exact words.

# Semantic Question 2
# | Feature                    | Keyword RAG                    | Semantic RAG |
# |----------------------------|--------------------------------|--------------|
# | What is compared?          | Exact word overlap             | Query and chunk embeddings |
# | What is retrieved?         | Full document                  | Most relevant text chunks |
# | Can it handle synonyms?    | No                             | Usually, by meaning |
# | Storage format             | Plain text dictionary          | Vector index/store |
# | Relevance score            | Number of overlapping keywords| Cosine/distance similarity |


# --- LlamaIndex ---
def run_llamaindex_warmups() -> None:
    """Run the semantic retrieval exercises when lesson resources are present."""
    docs_dir = Path("lessons/06_AI_augmentation/resources/brightleaf_pdfs")
    if not docs_dir.exists():
        print(f"\nSkipping LlamaIndex warmups: document directory not found: {docs_dir}")
        return
    if not os.getenv("OPENAI_API_KEY"):
        print("\nSkipping LlamaIndex warmups: OPENAI_API_KEY is not set.")
        return

    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    documents_loaded = SimpleDirectoryReader(str(docs_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents_loaded)
    query_engine = index.as_query_engine(similarity_top_k=3)

    # LlamaIndex Question 1
    questions = [
        "What employee benefits does BrightLeaf offer?",
        "What are BrightLeaf's security policies?",
    ]
    for question in questions:
        response = query_engine.query(question)
        print(f"\nQuestion: {question}\nAnswer: {response}")
        for node in response.source_nodes:
            print(f"Score: {node.score}; Chunk: {node.node.get_content()[:150]}")
    # Inspect source passages and answer tone for relevance, overconfidence, and
    # any surprising retrievals; actual observations depend on the course PDFs.

    # LlamaIndex Question 2
    question = questions[0]
    for top_k in (1, 5):
        response = index.as_query_engine(similarity_top_k=top_k).query(question)
        print(f"\nTop k={top_k}; answer: {response}")
        for node in response.source_nodes:
            print(f"Score: {node.score}")
    # More context can help when it adds evidence, but unrelated passages can
    # distract the model; compare the two answers and sources.

    # LlamaIndex Question 3
    hard_question = "What will BrightLeaf's electricity bill be next year?"
    response = query_engine.query(hard_question)
    print(f"\nQuestion: {hard_question}\nAnswer: {response}")
    for node in response.source_nodes:
        print(f"Score: {node.score}; Chunk: {node.node.get_content()}")
    # This forecast is likely outside the documents; abstention behavior and
    # retrieval quality show where a confidence threshold or clarification helps.

    # LlamaIndex Question 4
    faithfulness = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4o-mini"))
    relevancy = RelevancyEvaluator(llm=OpenAI(model="gpt-4o-mini"))
    for question in (questions[0], "What is the CEO's favorite movie?"):
        response = query_engine.query(question)
        faithfulness_result = faithfulness.evaluate_response(response=response)
        relevancy_result = relevancy.evaluate_response(query=question, response=response)
        print(f"\nEvaluation query: {question}")
        print("Faithfulness:", faithfulness_result.score, faithfulness_result.feedback)
        print("Relevancy:", relevancy_result.score, relevancy_result.feedback)
    # Faithfulness measures whether the answer is supported by retrieved context;
    # 1.0 indicates full support and 0.0 indicates no support. Relevancy measures
    # whether it addresses the question, which is distinct from factual grounding.
    # LLM-as-a-judge uses a capable model to assess nuanced answers that a simple
    # exact-match metric cannot fairly score. Compare scores across both queries.


if __name__ == "__main__":
    run_llamaindex_warmups()
