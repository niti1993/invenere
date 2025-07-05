import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rag_engine.rag_pipeline import RAGPipeline

# LangChain imports
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llama = Ollama(model="llama3:3.2")  # Update as needed

# Entity extraction prompt and chain
entity_extract_prompt = PromptTemplate(
    input_variables=["answer_text"],
    template=(
        "Extract the main methods, techniques, entities, or key concepts listed or described in the following answer. "
        "Return them as a comma-separated list (no explanations):\n\n"
        "{answer_text}\n\nList:"
    )
)
entity_extract_chain = LLMChain(llm=llama, prompt=entity_extract_prompt)

# Enhanced query rewriting prompt and chain
enhance_prompt = PromptTemplate(
    input_variables=["original_query", "recent_history", "entity_list"],
    template=(
        "Rewrite the user's question to be explicit and maximally useful for document retrieval. "
        "Use recent conversation to clarify the topic. "
        "If the question contains vague terms, use the supplied entity list in the rewritten query.\n\n"
        "Recent history:\n{recent_history}\n\nOriginal query: {original_query}\n\nEntity list: {entity_list}\n\nEnhanced query:"
    )
)
enhance_chain = LLMChain(llm=llama, prompt=enhance_prompt)

# Answer prompt and chain
answer_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an expert assistant. Use only the provided context to answer the user's question. "
        "If the answer is not in the context, say 'Not enough information.'\n\n"
        "Context:\n{context}\n\n"
        "User Query: {query}\n\n"
        "Answer:"
    )
)
answer_chain = LLMChain(llm=llama, prompt=answer_prompt)

pipeline = RAGPipeline()
print("✅ Vector store loaded from ChromaDB. Ready for search.")

history = []
MAX_TURNS = 10
summary = ""

def summarize_history(history):
    conversation = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    prompt = (
        "Summarize the following conversation briefly, capturing all key topics discussed so far:\n\n"
        f"{conversation}\n\nSummary:"
    )
    return llama(prompt)

def enhance_query_with_entities(original_query, history):
    recent = ""
    last_answer = ""
    if history:
        recent = "\n".join([f"Q: {q}\nA: {a}" for q, a in history[-3:]])
        last_answer = history[-1][1] if history else ""
    vague_terms = ["these", "them", "those", "the above", "such methods", "those methods"]
    needs_entity = any(term in original_query.lower() for term in vague_terms)
    entity_list = ""
    if needs_entity and last_answer:
        entity_list = entity_extract_chain.run(answer_text=last_answer).strip()
        print("Extracted entities for query rewriting:", entity_list)
    enhanced = enhance_chain.run(
        original_query=original_query,
        recent_history=recent,
        entity_list=entity_list
    ).strip()
    print("Enhanced query:", enhanced)
    return enhanced if enhanced else original_query

def print_metadata(src):
    if isinstance(src, dict):
        print("  - File:", src.get("source", "Unknown"))
        print("    Section:", src.get("heading", "None"))
        for k, v in src.items():
            if k not in ["source", "heading"]:
                print(f"    {k.capitalize()}: {v}")
    else:
        print("  -", src)

while True:
    try:
        query = input("\nEnter your search query (or type 'exit' to quit): ")
    except EOFError:
        print("\nExiting.")
        break

    if query.strip().lower() == "exit":
        break

    # --- 1. Enhance the query using LLaMA + LangChain ---
    enhanced_query = enhance_query_with_entities(query, history)
    print(f"\n🔍 Enhanced query: {enhanced_query}\n")

    # --- 2. RAG retrieval step ---
    retrieved_context, sources = pipeline.query(
        enhanced_query, return_sources=True, history=history
    )

    # --- 3. Answer generation using LangChain LLMChain ---
    response = answer_chain.run(context=retrieved_context, query=enhanced_query)

    print("\n🧠 LLaMA's Response:\n")
    print(response)
    print("\n📄 Source files and metadata used:")
    for src in sources:
        print_metadata(src)

    # --- 4. Update conversation history ---
    history.append((query, response))

    # --- 5. Summarize history if memory too long ---
    if len(history) > 2 * MAX_TURNS:
        summary = summarize_history(history[:-MAX_TURNS])
        history = history[-MAX_TURNS:]

    # --- 6. Print conversation history for debugging ---
    print("\nConversation History:")
    for i, (q, a) in enumerate(history[-MAX_TURNS:], 1):
        print(f"{i}. Q: {q}\n   A: {a}\n")
