from langgraph.graph import Graph

graph = Graph()
graph.add_node("llm",llm_node)
graph.add_edge("start","llm")
graph.add_edge("llm","end")
graph.run({"input":"Hello, how are you?"})