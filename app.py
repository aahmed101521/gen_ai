import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Health Tech AI", page_icon="🩺")
st.title("🩺 Medical AI (Now with Memory!)")

# --- 1. INITIALIZE MEMORY ---
# We use Streamlit's session_state to remember the chat across screen refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. THE BUFFER WINDOW LOGIC ---
# This function protects our token limit by ONLY grabbing the last K exchanges
def get_buffer_memory(k=3):
    # k=3 means 3 User messages + 3 AI messages = 6 total messages max
    recent_messages = st.session_state.messages[-(k*2):]
    
    memory_string = ""
    for msg in recent_messages:
        role = "User" if msg["role"] == "user" else "AI"
        memory_string += f"{role}: {msg['content']}\n"
    
    return memory_string

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./medical_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    llm = OllamaLLM(model="llama3.2")
    
    # --- 3. THE NEW PROMPT WITH MEMORY ---
    prompt_template = """
    You are an expert medical AI assistant. Use ONLY the following retrieved context to answer the question. 
    If you don't know the answer, say "I don't know."
    
    Here is the recent Chat History to give you context for follow-up questions:
    {chat_history}

    Context:
    {context}

    Question:
    {input}

    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

rag_chain = load_rag_pipeline()

# --- 4. THE CHAT INTERFACE ---
# Draw the existing chat history on the screen using chat bubbles
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input box at the bottom of the screen
if user_query := st.chat_input("Ask a medical question..."):
    
    # 1. Add user question to memory and screen
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Generate the AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Fetch our protected short-term memory
            current_memory = get_buffer_memory(k=3)
            
            # Pass BOTH the question and the memory to the chain
            response = rag_chain.invoke({
                "input": user_query,
                "chat_history": current_memory
            })
            
            answer = response["answer"]
            st.markdown(answer)
            
            # 3. Add AI answer to memory
            st.session_state.messages.append({"role": "assistant", "content": answer})