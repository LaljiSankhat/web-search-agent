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


class AgentState(TypedDict):
    # Chat
    userMessage: str
    chat_history: List[Union[HumanMessage, AIMessage]]

    # Research
    topic: Optional[str]
    sources: List[str]
    contents: List[str]
    summarized_content: str
    deepResearch: str

    # Refinement
    userInterest: Optional[str]
    userRelatedResearch: Optional[str]

    phase: Literal["init", "refine"]

def detect_phase(state: AgentState):
    if not state.get("deepResearch"):
        state["topic"] = state["userMessage"]
        state["phase"] = "init"
        return state
    else:
        state["userInterest"] = state["userMessage"]
        state["phase"] = "refine"
        return state

def router(state: AgentState):
    return state


def get_sources(state: AgentState):
    params = {
        "q": state['topic'],
        "api_key": os.getenv("SERP_API_KEY"),  
    }

    search = GoogleSearch(params)
    results = search.get_dict()          # parse to Python dict

    # l = [news['link'] for news in results['organic_results']]
    state['sources'] = [news['link'] for news in results.get('organic_results', [])[:1  ]]

    return state


def get_contents(state: AgentState):
    contents = []

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    for url in state["sources"]:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")

            text = " ".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 50
            )

            if text:
                contents.append(text)

        except Exception as e:
            print(f"Skipping {url} | Reason: {e}")

    state["contents"] = contents
    return state


def deep_think(state: AgentState):
    result = " ".join(state['contents'])

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

    # print(response.content)

    state['deepResearch'] = response.content
    return state

def human_approval(state: AgentState):
    decision = interrupt(
        {
            "question": "Do you want to research on specific topic ? ", 
            "options": ["yes", "no"]
        }
    )

    state["phase"] = "refine" if decision == "yes" else "init"
    return state


def refined_research(state: AgentState):

    response = deep_llm.invoke(
        user_interest_prompt.format_messages(
            topic=state['topic'],
            interest=state['userInterest'],
            combined_content=state['summarized_content']
        )
    )

    state['userRelatedResearch'] = response.content
    # print(state['userRelatedResearch'])
    return state




memory = MemorySaver()



graph = StateGraph(AgentState)

graph.add_node("detect_phase", detect_phase)
graph.add_node("router", router)
graph.add_node("get_sources", get_sources)
graph.add_node("get_contents", get_contents)
graph.add_node("deep_think", deep_think)
graph.add_node("refined_research", refined_research)
graph.add_node("approval", human_approval)

graph.add_edge(START, "detect_phase")
graph.add_edge("detect_phase", "router")

graph.add_conditional_edges(
    "router",
    lambda state: state["phase"],
    {
        "init": "get_sources",
        "refine": "refined_research",
    }
)

graph.add_conditional_edges(
    "approval",
    lambda state: state["phase"],
    {
        "refine": "refined_research",
        "init": END,
    }
)

graph.add_edge("get_sources", "get_contents")
graph.add_edge("get_contents", "deep_think")


graph.add_edge("deep_think", "approval") 
graph.add_edge("refined_research", END)


workflow = graph.compile(
    checkpointer=memory
)


# Configuration for the thread
config = {"configurable": {"thread_id": "conversation_1"}}


print("AI Research Agent: Hello! What topic would you like to research today?")

while True:
    # 1. Check if the graph is currently interrupted
    snapshot = workflow.get_state(config)
    current_history = snapshot.values.get("chat_history", [])

    if snapshot.interrupts:
        # Human Intervention
        research_output = snapshot.values.get("deepResearch")
        print("\n DEEP RESEARCH COMPLETED.... \n")
        print(research_output)

        current_history.append(AIMessage(content=research_output))
        
        # user_refinement = input("\nAI: What specific sub-topic should I analyze further? (yes or type 'exit'): ")
        
        # current_history.append(AIMessage(content="AI: What specific sub-topic should I analyze further? (or type 'exit')"))

        # current_history.append(HumanMessage(content=user_refinement))

        # if user_refinement.lower() in ["exit", "quit"]:
        #     break


        approval = input("\nDo you want specific research? (yes/no): ").strip().lower()

        workflow.invoke(
            Command(resume=approval),
            config
        )

        if approval != "yes":
            break  # graph will END cleanly

        # 2 refinement input
        user_refinement = input("\nEnter specific sub-topic: ")

        workflow.invoke(
            Command(
                resume=None,
                update={"userInterest": user_refinement, "chat_history": current_history}
            ),
            config
        )
            
        # Update the state 
        # workflow.update_state(config, {"userInterest": user_refinement, "chat_history": current_history})
        
        # Resume the graph (passing None tells it to proceed from the interrupt)
        # workflow.invoke(None, config)

        # workflow.invoke(
        #     Command(
        #         resume=user_refinement,
        #         update={"userInterest": user_refinement, "chat_history": current_history}
        #     ),
        #     config
        # )
        
        # After resuming, it runs 'refined_research' and then ends.
        final_snapshot = workflow.get_state(config)
        print("\n REFINED ANALYSIS ... \n")
        specific_research = final_snapshot.values.get("userRelatedResearch")
        print(specific_research)
        final_current_history = final_snapshot.values.get("chat_history", [])

        final_current_history.append(AIMessage(content=specific_research))
        
        # Clear specific topic for next loop or handle as needed
        # print("\nAI: Research session finished. You can start a new topic or exit.")

        # final_current_history.append(AIMessage(content="AI: Research session finished. You can start a new topic or exit."))

        workflow.update_state(config, {"chat_history": final_current_history})

        break
        
    else:
        # initial phase
        user_input = input("\nUser (Topic): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Start the graph with the initial topic
        workflow.invoke({"userMessage": user_input}, config)

        current_history.append(HumanMessage(content=user_input))

        workflow.update_state(config, {"chat_history": current_history})


# snapshot = workflow.get_state(config)
# print(json.dumps(snapshot, indent=2))

final_state = workflow.get_state(config).values
print(f"Total Messages: {len(final_state.get('chat_history', []))}")