from sentence_transformers import SentenceTransformer, util

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

#(converts text to embeddings) 
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast, free model
print("Creating embeddings for documents...")
doc_embeddings = model.encode(documents, convert_to_tensor=True)


def retrieve(question, top_k=2):
    question_embedding = model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, doc_embeddings)[0]
    top_results = scores.topk(top_k)
    print(f"\nQuestion: {question}")
    print("Most relevant documents found:")
    for score, idx in zip(top_results.values, top_results.indices):
        print(f"  [{score:.2f}] {documents[idx]}")

retrieve("What is RAG and why is it useful?")
retrieve("What are the risks of using AI in finance?")
retrieve("What programming language should I use for GenAI?")
retrieve("What does Nasdaq's AI team actually build?")