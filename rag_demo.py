# Mini RAG Demo — Alisa Biliavska
# Retrieval-Augmented Generation: find the most relevant document for any question
# No API key needed — runs fully locally using sentence-transformers

from sentence_transformers import SentenceTransformer, util

# --- STEP 1: Our "knowledge base" (documents we want to search over) ---
documents = [
    "Nasdaq is a global technology company serving the capital markets and other industries.",
    "Nasdaq's GenAI Platform team builds AI-powered tools to help financial professionals work smarter.",
    "RAG stands for Retrieval-Augmented Generation. It gives LLMs access to fresh, external data before generating an answer.",
    "Large Language Models like GPT are trained on massive text data and predict the next word in a sequence.",
    "Hallucination in AI means the model generates confident but incorrect or made-up information.",
    "Vector databases store text as numerical embeddings so you can search by meaning, not just keywords.",
    "Fine-tuning means retraining a model on specific data to permanently change its behavior.",
    "Python is the primary language for GenAI development, with libraries like LangChain and Hugging Face.",
]

# --- STEP 2: Load the model (converts text to embeddings) ---
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast, free model

# --- STEP 3: Convert all documents to embeddings (numbers that capture meaning) ---
print("Creating embeddings for documents...")
doc_embeddings = model.encode(documents, convert_to_tensor=True)

# --- STEP 4: Search function — find most relevant document for a question ---
def retrieve(question, top_k=2):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, doc_embeddings)[0]
    top_results = scores.topk(top_k)
    
    print(f"\nQuestion: {question}")
    print("Most relevant documents found:")
    for score, idx in zip(top_results.values, top_results.indices):
        print(f"  [{score:.2f}] {documents[idx]}")

# --- STEP 5: Try it out! ---
retrieve("What is RAG and why is it useful?")
retrieve("What are the risks of using AI in finance?")
retrieve("What programming language should I use for GenAI?")
retrieve("What does Nasdaq's AI team actually build?")