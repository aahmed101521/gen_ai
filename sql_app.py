import streamlit as st
import re
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure the Streamlit page
st.set_page_config(page_title="Store AI Manager", page_icon="🛒")
st.title("🛒 Electronics Store SQL Agent")
st.markdown("Ask natural language questions about the inventory and watch the AI write the SQL!")

@st.cache_resource
def load_sql_pipeline():
    # 1. Connect to DB and LLM
    db = SQLDatabase.from_uri("sqlite:///electronics_store.db")
    llm = OllamaLLM(model="llama3.2")
    execute_query_tool = QuerySQLDatabaseTool(db=db)

    # 2. Few-Shot Examples
    examples = """
    Question: How many white Apple smartphones do we have in stock?
    SQLQuery: SELECT stock_quantity FROM inventory WHERE color = 'White' AND brand = 'Apple' AND category = 'Smartphone';

    Question: What is the total inventory value of all Dell laptops?
    SQLQuery: SELECT SUM(price * stock_quantity) FROM inventory WHERE brand = 'Dell' AND category = 'Laptop';
    """

    # 3. Aggressive Prompting
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

    write_query_chain = create_sql_query_chain(llm, db, custom_prompt)

    # 4. The Regex Sanitizer
    def extract_and_run_sql(vars):
        raw_sql = vars["query"]
        match = re.search(r'(?i)(SELECT.*?;)', raw_sql, re.DOTALL)
        
        if match:
            cleaned_sql = match.group(1).strip()
        else:
            cleaned_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
            if "SQLQuery:" in cleaned_sql:
                cleaned_sql = cleaned_sql.split("SQLQuery:")[-1].strip()
        
        # Save the pure SQL to session state so we can show it on the UI
        st.session_state.last_sql = cleaned_sql
        return execute_query_tool.invoke(cleaned_sql)

    # 5. The English Translator
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, and SQL result, answer the user question in a polite, natural English sentence.
        
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
    )
    answer_chain = answer_prompt | llm | StrOutputParser()

    # 6. Assemble the Chain
    chain = (
        RunnablePassthrough.assign(query=write_query_chain).assign(
            result=extract_and_run_sql
        )
        | answer_chain
    )
    return chain

# Setup session state for the UI
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""

# Load the pipeline
chain = load_sql_pipeline()

# User Interface
st.divider()
user_query = st.text_input("Ask the Store Manager:")

if user_query:
    with st.spinner("Writing SQL and querying database..."):
        final_answer = chain.invoke({"question": user_query})
        
        st.subheader("Answer:")
        st.success(final_answer)
        
        with st.expander("🔍 View the Executed SQL"):
            st.code(st.session_state.last_sql, language="sql")