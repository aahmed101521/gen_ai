from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# --- THE FIX FOR LANGCHAIN v1.0+ ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

print("1. Waking up the Embedding Model...")
# Must be the exact same model we used to create the database!
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("2. Connecting to the Vault (ChromaDB)...")
vector_db = Chroma(persist_directory="./medical_db", embedding_function=embeddings)

# We set up a "Retriever" to fetch the top 3 most mathematically relevant chunks (k=3)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

print("3. Connecting to Llama 3.2...")
llm = OllamaLLM(model="llama3.2")

print("4. Forging the Chain...\n")
# This is our custom prompt. We tell the AI strictly to use our provided context.
prompt_template = """
You are an expert medical AI assistant. Use ONLY the following retrieved context to answer the question. 
If you don't know the answer based on the context, just say "I don't know." Do not make anything up.

Context:
{context}

Question:
{input}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# LangChain magic: Link the AI, the Prompt, and the Database together
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

print("✅ Bot is online! Type 'exit' to quit.\n")
print("-" * 50)

# Create a continuous loop so you can chat with it
while True:
    user_query = input("Ask the Medical Bot: ")
    
    if user_query.lower() == 'exit':
        print("Shutting down...")
        break
        
    # Send the question through the RAG pipeline
    response = rag_chain.invoke({"input": user_query})
    
    # --- 🔍 THE X-RAY DEBUGGER ---
    # This reveals exactly what the Vector Database handed to Llama 3.2
    retrieved_chunks = response.get("context", [])
    print(f"\n[🔍 DEBUG] The Vault retrieved {len(retrieved_chunks)} chunks.")
    
    if len(retrieved_chunks) > 0:
        print("[🔍 DEBUG] Here is the very first chunk the AI saw:")
        print("---")
        print(retrieved_chunks[0].page_content[:300] + "...\n---") # Print the first 300 characters
    
    print("\n🤖 Answer:")
    print(response["answer"])
    print("-" * 50)