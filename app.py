import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Configure the web page
st.set_page_config(page_title="Health Tech AI", page_icon="🩺")
st.title("🩺 Private Medical AI Assistant")
st.markdown("Ask questions based on our secure medical database.")

# --- THE CACHE ---
# We use @st.cache_resource so the app doesn't reboot the AI every time you type a letter!
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./medical_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    llm = OllamaLLM(model="llama3.2")
    
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
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Boot up the pipeline
rag_chain = load_rag_pipeline()

# --- THE USER INTERFACE ---
st.divider()
user_query = st.text_input("What would you like to know?")

if user_query:
    # Show a loading spinner while Llama 3.2 thinks
    with st.spinner("Analyzing medical database..."):
        response = rag_chain.invoke({"input": user_query})
        
        st.subheader("Answer:")
        st.write(response["answer"])
        
        # Pro-feature: Let's show the user exactly where the bot got its info!
        with st.expander("🔍 View Retrieved Context (Debugger)"):
            for i, chunk in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.caption(chunk.page_content)