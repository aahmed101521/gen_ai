Key Features:

Web Scraping: Uses BeautifulSoup to ingest unstructured medical knowledge.

Vector Database: Uses ChromaDB and HuggingFace's all-MiniLM-L6-v2 embeddings for fast semantic search.

Anti-Hallucination: Strictly prompts Llama 3.2 to only use retrieved context, or admit "I don't know."

How to run:

Build the database: python ingest.py

Run the Web App: streamlit run app.py

📁 Project 2: Text-to-SQL Agent

An intelligent agent that connects to a relational database. It translates natural English questions into executable SQL queries, runs them locally, and translates the data back into human language.

Key Features:

Automated Schema Reading: Uses SQLAlchemy to dynamically read table and column structures.

Few-Shot Learning: Injects specific business-logic examples into the prompt to guide query generation.

Regex Sanitization: A robust output parser that isolates raw SQL from the LLM's conversational text.

How to run:

Generate the SQLite Database: python setup_db.py

Run the Web App: streamlit run sql_app.py

⚙️ Tech Stack

LLM: Llama 3.2 (via Ollama)

Framework: LangChain 1.0+

Vector Store: ChromaDB

Database: SQLite

Frontend: Streamlit