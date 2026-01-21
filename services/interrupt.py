from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    task: str
    plan: str
    approved: bool

def generate_plan(state: AgentState):
    task = state["task"]
    print("planner is running")
    
    plan = f"""
    1. Search web for {task}
    2. Extract content
    3. Summarize findings
    """

    return {"plan": plan}

def human_approval(state: AgentState):
    print("\n approval is running")
    response = interrupt(
        {
            "question": "Do you approve this plan?",
            "plan": state["plan"],
            "options": ["yes", "no"]
        }
    )

    if response == "yes":
        return {"approved": True}
    else:
        return {"approved": False}

def execute_plan(state: AgentState):
    print("\n Plan approved. Executing...")
    return {}


graph = StateGraph(AgentState)

graph.add_node("planner", generate_plan)
graph.add_node("approval", human_approval)
graph.add_node("execute", execute_plan)

graph.add_edge(START, "planner")
graph.add_edge("planner", "approval")
graph.add_edge("approval", "execute")
graph.add_edge("execute", END)


config = {"configurable": {"thread_id": "c_1"}}
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 1️ Start graph (hits interrupt)
app.invoke(
    {
        "task": "LangGraph vs LangChain comparison",
        "approved": False
    },
    config
)

# 2️ Terminal input
human_input = input("Do you want to continue? (yes or no): ").strip().lower()

# 3️ Resume interrupt CORRECTLY

app.invoke(
    Command(
        resume="yes",                  # value returned by interrupt()
        update={"approved": True}       # state updates (optional)
    ),
    config
)