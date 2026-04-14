from langchain_ollama import OllamaLLM

# 1. Initialize the connection
llm = OllamaLLM(model="llama3.2")

# 2. Define a prompt
prompt = "Explain the offside rule in football in exactly one simple sentence."

# 3. Call the model
print(f"Sending prompt to Llama 3.2...\n")
response = llm.invoke(prompt)

# 4. Print the output
print("Response:")
print(response)