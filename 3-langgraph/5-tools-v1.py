from typing import TypedDict, Optional,List 
from langgraph.graph import  START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated
import numexpr
from langchain_core.messages import HumanMessage


  # take environment variables from .env

load_dotenv()

@tool
def calculator(query: str) -> str:
    """
    A Simple Calculator tool. Input should be a mathematical expression.
    """
    return str(numexpr.evaluate(query))

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, calculator]

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.1).bind_tools(tools)

class State(TypedDict):
    """
    messages: user input messages
    """
    messages: Annotated[list, add_messages]

def llm_node(state: State) -> State:
    print("Messages passed to LLM:")
    print(state["messages"])
    result = llm_gemini.invoke(state["messages"])
    print("Intermediate Result from LLM:")
    print(result)
    return {"messages":[result]}

builder = StateGraph(State)
builder.add_node("model", llm_node)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START,"model")
builder.add_conditional_edges("model",tools_condition)
builder.add_edge("tools","model")
builder.add_edge("model",END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="basic-agent.png")

input = {
    "messages": [HumanMessage(content="calculate 123*45.")]
}

input = {
    "messages": [HumanMessage(content="Calculate the Fixed deposit interest for 100000 INR for 1 year at the current RBI repo rate?")]
}
result = graph.invoke(input)
print("Final Result:")
print(result)