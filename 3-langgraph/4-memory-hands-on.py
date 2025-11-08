from typing import TypedDict, Optional,List 
from langgraph.graph import  START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
  # take environment variables from .env

load_dotenv()


llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                temperature=0.5,
                                max_output_tokens=2048)

class GraphState(TypedDict):
    """
    input: latest user input
    summary: rolling summary
    messages: Ordered chat history
    """
    messages: List[BaseMessage]
    input: str 
    summary: Optional[str]

# ingest_user 

def ingest_user(state: GraphState) -> GraphState: 
    messages = list(state.get("messages",[]))
    user_text = state["input"]
    messages.append(HumanMessage(content=user_text))
    # print("Ingested user message:")
    # pprint.pp(messages)
    return {"messages":messages}

# chat node
def chat(state: GraphState) -> GraphState:
    messages = list(state.get("messages",[]))
    summary = state.get("summary","")

    effective_message: List[BaseMessage] = [] # previous summary as system message, and it should be prepended to the messages
    if summary:
        effective_message.append(SystemMessage(content=f"This is the current conversation summary: {summary}"))
    effective_message.extend(messages)

    response = llm_gemini.invoke(effective_message)
    messages.append(AIMessage(content=response.content))
    return {"messages":messages}

# update summary node
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at creating concise summaries of conversations."),
    ("human",
     "Given the following conversation history, provide a concise summary focusing "
     "on key facts, preferences, and decisions:\n\n{conversation_history}")
])
THRESHOLD = 2  # max number of messages before summarization

def summarize_if_long(state: GraphState) -> GraphState:
    messages = list(state.get("messages",[]))
    summary = state.get("summary","")

    # check if messages length exceeds threshold
    if len(messages) < THRESHOLD: 
        return {"summary": summary}  # no update

    # generate new summary
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    prompt = SUMMARY_PROMPT.format_prompt(conversation_history=conversation_text).to_messages()
    response = llm_gemini.invoke(prompt) # generate the conversation summary
    new_summary = response.content

    return {"summary": new_summary}

# build the graph
builder = StateGraph(GraphState)
builder.add_node("ingest_user", ingest_user)
builder.add_node("chat", chat)
builder.add_node("summarize_if_long", summarize_if_long)

builder.add_edge(START, "ingest_user")
builder.add_edge("ingest_user", "chat")
builder.add_edge("chat", "summarize_if_long")
builder.add_edge("summarize_if_long", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

def run_turn(user_input: str, thread_id: str):
    # initial state
    result = app.invoke(
        {"input": user_input},
         config = {"configurable":{"thread_id": thread_id}}
    )
    return result

if __name__ == "__main__":
    thread_id = "user-5678"
    print("Turn 1 : Hi, I am planning a 5 day trip to Penang in April. I love Nasi Lemak and historical sites.")
    s1 = run_turn("Hi, I am planning a 5 day trip to Penang in April. I love Nasi Lemak and historical sites.", thread_id)

    print("After Turn 1:")
    print(s1["messages"][-1].content)

    # Turn 2
    print("Turn 2 : Can you suggest an itinerary for me?")
    s2 = run_turn("Can you suggest an itinerary for me?", thread_id)
    print("\nAfter Turn 2:")
    print(s2["messages"][-1].content)
    print("Turn 3 : Also suggest some other local delicacies I should try.")
    s3 = run_turn("Also suggest some other local delicacies I should try.", thread_id)
    print("\nAfter Turn 3:")
    print(s3["messages"][-1].content)

    # show the persisted storage
    print("\nPersisted State Snapshot:")
    print("summary:",s3.get("summary"))
