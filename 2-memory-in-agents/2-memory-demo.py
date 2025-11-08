from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import argparse
  # take environment variables from .env

load_dotenv()


llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assitant. Answer as briefly as possible."),
    ("user", "{input}?")
])

memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(llm=llm_gemini, memory=memory, verbose=False)

print("Full History Memory Demo. Type 'exit' to quit.")
print("type /show to print the raw conversation memory.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Exiting the demo.")
        break
    if user_input.lower() == '/show':
        print("\n--- Conversation Memory ---")
        for msg in memory.chat_memory.messages:
            print(f"{msg.type}: {msg.content}")
        print("---------------------------\n")
        continue

    response = chain.invoke(input=user_input)
    print(f"Bot:{response.content}\n")