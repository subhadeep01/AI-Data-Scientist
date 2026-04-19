from langgraph.graph import StateGraph, END
from agents.preprocessing_agent import preprocessing_agent
from agents.modeling_agent import modeling_agent
from agents.evaluation_agent import evaluation_agent
from agents.explainability_agent import explainability_agent

def build_graph(llm, state_type):
    builder = StateGraph(state_type)

    builder.add_node("preprocessing", preprocessing_agent)
    builder.add_node("modeling", modeling_agent)
    builder.add_node("explainability", explainability_agent)

    # wrap evaluation with llm
    builder.add_node("evaluation", lambda state: evaluation_agent(state, llm))

    builder.set_entry_point("preprocessing")

    builder.add_edge("preprocessing", "modeling")
    builder.add_edge("modeling", "evaluation")
    builder.add_edge("evaluation", "explainability")
    builder.add_edge("explainability", END)

    return builder.compile()