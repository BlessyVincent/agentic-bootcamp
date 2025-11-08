from typing import Annotated, TypedDict
import os
from wikipedia import summary as wiki_summary
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# State Definition
class WorkflowState(TypedDict, total=False):
    query: str
    context: Annotated[str, 'text fetched from wikipedia']
    summary: Annotated[str, 'LLM-generated summary']
    summary_formatted: Annotated[str, 'formatted bullet points']  # For Extension 7A

# Define Your Tools
# 4.1 Wikipedia Tool
def wikipedia_tool(topic: str) -> str:
    try:
        return wiki_summary(topic, sentences=8, auto_suggest=False, redirect=True)
    except Exception as e:
        return f'Could not fetch from Wikipedia for {topic}. Error: {e}'

# 4.2 Gemini Summarizer Tool
load_dotenv()
gemini = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.3
)

def summarize_with_gemini_tool(text: str) -> str:
    if not text or 'Could not fetch' in text:
        return 'No valid context to summarize.'
    prompt = PromptTemplate.from_template('''You are a helpful assistant that summarizes technical/contextual text.
Summarize the following text in 4–6 lines, keeping product names intact.

{context}

Summary:''')
    chain = prompt | gemini
    out = chain.invoke({'context': text})
    return out.content

# Extension 7A - Formatter Agent Tool
def format_bullet_points(query: str, summary: str) -> str:
    prompt = PromptTemplate.from_template('''Convert this summary into 3-5 clear bullet points.
Keep important terms, numbers, and names intact.

Summary: {summary}

Format as:
Summary for {query}:
- Point 1
- Point 2
- Point 3''')
    
    chain = prompt | gemini
    out = chain.invoke({'summary': summary, 'query': query})
    # Extension 7C - Add source attribution
    return f"{out.content}\n\nSource: Wikipedia"

# 5.1 Research Agent
def research_agent(state: WorkflowState) -> WorkflowState:
    query = (state.get('query') or '').strip()
    if not query:
        state['context'] = 'No query provided.'
        return state

    context = wikipedia_tool(query)
    state['context'] = context
    return state

# 5.2 Summary Agent
def summary_agent(state: WorkflowState) -> WorkflowState:
    context = state.get('context') or ''
    summary = summarize_with_gemini_tool(context)
    state['summary'] = summary
    return state

# Extension 7A - Formatter Agent
def formatter_agent(state: WorkflowState) -> WorkflowState:
    query = state.get('query') or ''
    summary = state.get('summary') or ''
    formatted = format_bullet_points(query, summary)
    state['summary_formatted'] = formatted
    return state

# 6. Wire It with LangGraph
def build_workflow():
    graph = StateGraph(WorkflowState)
    # Add nodes
    graph.add_node('research_agent', research_agent)
    graph.add_node('summary_agent', summary_agent)
    graph.add_node('formatter_agent', formatter_agent)
    
    # Add edges
    graph.add_edge(START, 'research_agent')
    graph.add_edge('research_agent', 'summary_agent')
    graph.add_edge('summary_agent', 'formatter_agent')  # Extension 7A flow
    graph.add_edge('formatter_agent', END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

if __name__ == '__main__':
    workflow = build_workflow()
    
    # Extension 7B - Interactive Query
    user_query = input("Enter a topic to summarize: ")
    initial_state = {'query': user_query}
    
    final_state = workflow.invoke(initial_state, config={'configurable': {'thread_id': 'demo_user_1'}})
    
    # Print results
    print('\nQuery:', initial_state['query'])
    print('\nContext:\n', final_state.get('context'))
    print('\nSummary:\n', final_state.get('summary'))
    print('\nFormatted Summary:\n', final_state.get('summary_formatted'))