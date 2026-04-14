from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Scraping Medical/Tech Articles...")
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare",
    "https://en.wikipedia.org/wiki/Telehealth"
]

# WebBaseLoader is a fast, standard scraper built into LangChain
loader = WebBaseLoader(urls)
documents = loader.load()
print(f"Loaded {len(documents)} massive web pages.\n")

print("2. Chopping text into chunks...")
# We use a 1000 character chunk size, with a 100 character overlap so we don't slice sentences in half!
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} individual overlapping chunks.\n")

print("3. Downloading Embedding Model & Generating Vectors...")
# Instead of paying OpenAI, we use a lightweight, powerful open-source model from HuggingFace
# Note: The first time you run this, it will take a minute to download the model (~80MB)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("4. Saving to ChromaDB Vector Database...")
# We save the database to a local folder called "medical_db" so we can query it later
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./medical_db"
)

print(f"✅ Success! {len(chunks)} chunks mathematically embedded and saved to ./medical_db")