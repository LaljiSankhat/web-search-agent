from typing import TypedDict, List, Optional, Literal, Union
from unittest import result
from langgraph.graph import StateGraph, START, END
from serpapi import GoogleSearch
import os
import json
import requests
from bs4 import BeautifulSoup
from services.llms import map_llm, deep_llm
from services.prompts import deep_think_prompt, user_interest_prompt, map_prompt
from services.text_splitter import split_text_into_chunks
import time
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import interrupt, Command
from services.tavily_search import tavily_client


# search = TavilySearch()

class AgentState(TypedDict):
    # Chat
    userMessage: str

    # Research
    topic: Optional[str]
    # sources: List[str]
    contents: List[str]
    summarized_content: str
    deepResearch: str

    # Refinement
    userInterest: Optional[str]
    userRelatedResearch: Optional[str]

    satisfied: Optional[bool]



def get_contents(state: AgentState):
    topic_user = state["userMessage"]
    state["topic"] = topic_user

    response = tavily_client.search(
        query=topic_user,
        search_depth="advanced",      
        include_raw_content=False,    
        # max_results=5
    )

    documents = []

    for r in response["results"]:
        if r.get("content"):
            documents.append(
                f"{r['content']}"
            )

    combined_content = "\n\n".join(documents)

    state["contents"] = combined_content[:]

    return state



def deep_think(state: AgentState):
    result = " ".join(state['contents'])
    state["summarized_content"] = ""


    chunks = split_text_into_chunks(result)

    chunk_summaries = []

    print(f"Total chunks to analyse {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1}...")
        chunk_response = map_llm.invoke(map_prompt.format(chunk_content=chunk))
        chunk_summaries.append(chunk_response.content)
        time.sleep(1)

    # final_combined_content = "\n\n".join(chunk_summaries)
    state['summarized_content'] = "\n\n".join(chunk_summaries)

    print("Starting final Deep Think analysis...")

    # 4. Final Deep Think Call
    response = deep_llm.invoke(
        deep_think_prompt.format_messages(
            topic=state['topic'],
            combined_content=state['summarized_content']
        )
    )

    print(response.content)

    state['deepResearch'] = response.content
    return state

def human_approval(state: AgentState):
    decision = interrupt(
        {
            "question": "Do you want to research on specific topic ? ", 
            "options": ["yes", "no"]
        }
    )

    if decision == "no":
        return {"userInterest": None}
    return {} 


def satisfaction(state: AgentState):
    decision = interrupt(
        {"question": "Are you satisfied?", "options": ["yes", "no"]}
    )
    return {"satisfied": decision == "yes"}


def refined_research(state: AgentState):

    response = deep_llm.invoke(
        user_interest_prompt.format_messages(
            topic=state['topic'],
            interest=state['userInterest'],
            combined_content=state['summarized_content']
        )
    )

    state['userRelatedResearch'] = response.content
    print(state['userRelatedResearch'])
    return state




memory = MemorySaver()



graph = StateGraph(AgentState)


# graph.add_node("get_sources", get_sources)
graph.add_node("get_contents", get_contents)
graph.add_node("deep_think", deep_think)
graph.add_node("refined_research", refined_research)
graph.add_node("approval", human_approval)
graph.add_node("satisfied", satisfaction)



# graph.add_edge(START, "get_sources")
graph.add_edge(START, "get_contents")
graph.add_edge("get_contents", "deep_think")
graph.add_edge("deep_think", "approval") 



graph.add_conditional_edges(
    "approval",
    lambda state: state["userInterest"] is not None,
    {
        True: "refined_research",
        False: END,
    }
)

graph.add_edge("refined_research", "satisfied")

graph.add_conditional_edges(
    "satisfied",
    lambda state: state["satisfied"],
    {
        True: END,                   
        False: "approval", 
    }
)


workflow = graph.compile(
    checkpointer=memory
)


# Configuration for the thread
config = {"configurable": {"thread_id": "conversation_1"}}


print("AI Research Agent: Hello! What topic would you like to research today?")

# print("AI Research Agent Ready")

while True:
    user_topic = input("\nEnter topic (or exit): ").strip()
    if user_topic.lower() in ["exit", "quit"]:
        break

    workflow.invoke(
        {
            "userMessage": user_topic,
            "topic": user_topic,
            "contents": [],
            "summarized_content": "",
            "deepResearch": "",
            "userInterest": None,
            "userRelatedResearch": None,
            "satisfied": None
        },
        config
    )


    while True:
        snapshot = workflow.get_state(config)

        if not snapshot.interrupts:
            break

        interrupt_data = snapshot.interrupts[0].value
        print("\n", interrupt_data["question"])
        print("Options:", interrupt_data["options"])

        user_input = input("> ").strip().lower()

        # If refinement needed, ask sub-topic
        if interrupt_data["question"].startswith("Do you want to research") and user_input == "yes":
            sub_topic = input("\nEnter specific sub-topic: ").strip()
            workflow.invoke(
                Command(resume=user_input, update={"userInterest": sub_topic}),
                config
            )
        else:
            workflow.invoke(Command(resume=user_input), config)

    print("\n Research cycle completed ...")


# final_state = workflow.get_state(config).values
# print(f"Total Messages: {len(final_state.get('chat_history', []))}")