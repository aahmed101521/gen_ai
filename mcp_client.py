import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

async def main():
    # 1. Start the local LLM
    print("1. Waking up Llama 3.2...")
    model = ChatOllama(model="llama3.2")

    # 2. Connect to the MCP Server
    print("2. Connecting to the MCP HR Server via stdio (USB-C)...")
    
    # We use MultiServerMCPClient because it automatically handles the complex 
    # stdio subprocess management and tool discovery in the background!
    client = MultiServerMCPClient({
        "leave_manager": {
            "command": "python",
            "args": ["mcp_hr_server.py"], # Pointing to the server script from the Canvas
            "transport": "stdio",
        }
    })

    # 3. Dynamically load the tools! Notice we didn't write ANY glue code here.
    tools = await client.get_tools()
    print(f"✅ Successfully loaded {len(tools)} tools dynamically from the server!")

    # 4. Assemble the Agent
    # We use LangGraph's prebuilt ReAct agent which handles tool-calling natively
    agent = create_react_agent(model, tools)

    # 5. Let's test a complex, multi-step command!
    question = "What is the leave balance for E001? Also, apply for a leave for them on 2026-12-25."
    print(f"\nUser: {question}\n")

    print("--- 🧠 AGENT INTERNAL MONOLOGUE ---")
    
    # Run the agent asynchronously
    response = await agent.ainvoke({"messages": [("user", question)]})

    print("\n--- 🤖 FINAL OUTPUT ---")
    # The final message in the state contains the AI's natural language response
    print(response["messages"][-1].content)

if __name__ == "__main__":
    # MCP Clients run asynchronously, so we wrap the main loop in asyncio
    asyncio.run(main())