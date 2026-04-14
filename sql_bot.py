from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_sql_query_chain
# Fix #1: The updated Database Tool import
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re

# 1. Connect to the DB and LLM
db = SQLDatabase.from_uri("sqlite:///electronics_store.db")
llm = OllamaLLM(model="llama3.2")

# 2. The Cheat Sheet (Few-Shot Examples)
examples = """
Question: How many white Apple smartphones do we have in stock?
SQLQuery: SELECT stock_quantity FROM inventory WHERE color = 'White' AND brand = 'Apple' AND category = 'Smartphone';

Question: What is the total inventory value of all Dell laptops?
SQLQuery: SELECT SUM(price * stock_quantity) FROM inventory WHERE brand = 'Dell' AND category = 'Laptop';
"""


# 3. Forging a Custom Prompt
custom_prompt = PromptTemplate.from_template(
    """You are a {dialect} expert. Given an input question, create a syntactically correct {dialect} query to run.
    Unless otherwise specified, do not return more than {top_k} results.
    
    CRITICAL INSTRUCTION: You must output ONLY the raw SQL query. Do not include any explanations, markdown formatting, or conversational text. Start directly with SELECT and end with a semicolon.
    
    Here is the table schema:
    {table_info}
    
    Here are some examples of perfect queries:
    """ + examples + """
    
    Question: {input}
    SQLQuery:"""
)

# 4. Building the "Write the SQL" Chain
write_query_chain = create_sql_query_chain(llm, db, custom_prompt)

# 5. Building the "Execute the SQL" Tool
execute_query_tool = QuerySQLDatabaseTool(db=db)

# --- THE FIX #2: The SQL Sanitizer ---
# This intercepts the chatty LLM and isolates just the pure SQL command
# --- THE FIX #3: The REGEX SQL Sanitizer ---
def extract_and_run_sql(vars):
    raw_sql = vars["query"]
    print(f"\n[🧠 RAW LLM OUTPUT] {raw_sql}") # Let's see exactly what it tried to do
    
    # Use Regex to hunt down the exact block starting with SELECT and ending with ;
    match = re.search(r'(?i)(SELECT.*?;)', raw_sql, re.DOTALL)
    
    if match:
        cleaned_sql = match.group(1).strip()
    else:
        # Fallback if it somehow forgot the semicolon
        cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        if "SQLQuery:" in cleaned_sql:
            cleaned_sql = cleaned_sql.split("SQLQuery:")[-1].strip()
            
    print(f"\n[🔍 DEBUG] Pure SQL sent to database: {cleaned_sql}")
    
    # Execute the cleaned SQL
    return execute_query_tool.invoke(cleaned_sql)


# 6. Building the "Translate to English" Chain
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question in a polite, natural English sentence.
    
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)
answer_chain = answer_prompt | llm | StrOutputParser()

# 7. The Grand Master Chain
chain = (
    RunnablePassthrough.assign(query=write_query_chain).assign(
        result=extract_and_run_sql # We inject our sanitizer here!
    )
    | answer_chain
)

print("✅ SQL Agent is Online and Armed with Few-Shot Learning.\n")

# Let's test it!
question = "How many white Apple smartphones do we have in stock?"
print(f"User: {question}")

final_answer = chain.invoke({"question": question})
print(f"🤖 AI: {final_answer}")