from langchain_ollama import OllamaLLM
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper

print("1. Equipping Tools (Wikipedia & Calculator)...")
wiki = WikipediaAPIWrapper()

# We can easily create our own custom tools using standard Python!
def simple_calculator(equation_string):
    try:
        return str(eval(equation_string))
    except:
        return "Error: Invalid math expression."

# We hand the tools to LangChain with strict descriptions so the AI knows WHEN to use them
tools = [
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Use this to search for facts about people, places, history, or current events. Input should be a search term."
    ),
    Tool(
        name="Calculator",
        func=simple_calculator,
        description="Use this to solve math problems. Input must be a valid mathematical expression (e.g., '2024 - 1990')."
    )
]

print("2. Waking up Llama 3.2...")
llm = OllamaLLM(model="llama3.2")

print("3. Forging the ReAct Prompt...")
# This is the industry-standard ReAct prompt. Notice how we force it to follow a strict loop!
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following exact format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, MUST be exactly one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL FORMATTING RULES:
1. You MUST use 'Action Input:' after 'Action:'.
2. Your 'Action:' MUST be exactly one of: [{tool_names}]. Do not make up your own action names.

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# 4. Assemble the Agent
agent = create_react_agent(llm, tools, react_prompt)

# verbose=True is the magic! It forces LangChain to print the AI's internal monologue to the terminal
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

print("✅ General ReAct Agent is Online!\n")

# Let's test it with a complex, multi-step question!
question = "How old was Albert Einstein when he died? Multiply that number by 10."
print(f"User: {question}\n")

print("--- 🧠 AGENT INTERNAL MONOLOGUE ---")
response = agent_executor.invoke({"input": question})

print("\n--- 🤖 FINAL OUTPUT ---")
print(response["output"])