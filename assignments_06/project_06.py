"""Week 6 mini-project: Groundwork Coffee document Q&A assistant."""

import os
from pathlib import Path

from dotenv import load_dotenv

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Paths are relative to the repository root, where this script should be run.
docs_dir = Path("lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is not set in .env"

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Step 2: Load the documents.
documents = SimpleDirectoryReader(str(docs_dir)).load_data()
print(f"Loaded {len(documents)} documents.")
for document in documents:
    print(document.metadata.get("file_name", "Unknown file"))

# Step 3: Build the semantic index and query engine.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully. Ready to answer questions.")


def print_answer(question: str, *, all_sources: bool = False) -> None:
    """Print an answer and its most relevant source evidence."""
    response = query_engine.query(question)
    print(f"\nQuestion: {question}\nAnswer: {response}")
    nodes = response.source_nodes if all_sources else response.source_nodes[:1]
    for source in nodes:
        name = source.node.metadata.get("file_name", "Unknown file")
        print(f"Source: {name}; score: {source.score}")
        print("Chunk:", source.node.get_content()[:200])


# Step 4: Ask the five required questions.
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for question in questions:
    print_answer(question)
# Review these outputs against the retrieved passages. The source excerpts make
# it possible to spot unsupported details and surprising answers.

# Step 5: Ask a question whose answer is unlikely to be in the business files.
print_answer("What was Groundwork's total revenue last quarter?", all_sources=True)
# This asks for private financial data likely absent from public-facing shop
# documents. Check whether retrieval finds unrelated text and whether the model
# guesses anyway. A production assistant could improve with abstention thresholds,
# clearer source coverage, and a request for clarification when evidence is weak.

# Step 6: Reflection
# This implementation uses a few lines to load documents, configure embeddings,
# create an index, and make a query engine. Writing extraction, chunking,
# embedding, storage, and similarity search directly would take substantially
# more code; a framework handles those common details and lets us focus on the
# data and evaluation. A similar system could answer employee questions from an
# up-to-date HR handbook. RAG still cannot prevent errors in source documents or
# guarantee that the model faithfully uses retrieved evidence in every answer.
