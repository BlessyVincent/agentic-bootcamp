from typing import TypedDict
from langgraph.graph import  START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
  # take environment variables from .env

load_dotenv()


llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048)

def llm_node(state: dict) -> dict:
    user_input = state.get("prompt")
    # user_input = state["input"]
    response = llm_gemini.invoke(user_input)
    return {"response": response.content}

class GraphState(TypedDict):
    prompt: str
    response: str
    
graph = StateGraph(GraphState)
graph.add_node("llm", llm_node)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

#input_state = {"prompt": "Explain Langraph in one Sentence."}
input_state = {"prompt": "Explain Langraph in one Sentence."}
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-1234"}}


result = app.invoke(input_state,config=config)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Final Output:", result["response"])

snapshot = app.get_state(config) # state snapshot
print("Snapshot Values:")
print(snapshot.values)

