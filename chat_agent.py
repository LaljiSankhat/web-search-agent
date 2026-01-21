from typing import TypedDict, List, Optional, Literal
from unittest import result
from langgraph.graph import StateGraph, START, END
from serpapi import GoogleSearch
import os
import json
import requests
from bs4 import BeautifulSoup
from services.deep_think import map_llm, deep_llm
from services.prompts import deep_think_prompt, user_interest_prompt, map_prompt
from services.text_splitter import split_text_into_chunks
import time


class AgentState(TypedDict):
    # Chat
    userMessage: str
    chat_history: List[str]

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
            topic="Python and ML",
            combined_content=state['summarized_content']
        )
    )

    # print(response.content)

    state['deepResearch'] = response.content
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


graph = StateGraph(AgentState)

graph.add_node("detect_phase", detect_phase)
graph.add_node("router", router)

graph.add_node("get_sources", get_sources)
graph.add_node("get_contents", get_contents)
graph.add_node("deep_think", deep_think)
graph.add_node("refined_research", refined_research)

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

graph.add_edge("get_sources", "get_contents")
graph.add_edge("get_contents", "deep_think")
graph.add_edge("deep_think", END)

graph.add_edge("refined_research", END)

workflow = graph.compile()


state = {
    "chat_history": [],
    "sources": [],
    "contents": [],
    "summarized_content": "",
    "deepResearch": "",
    "userRelatedResearch": "",
}

print("AI Research Agent: Hello! What topic would you like to research today?")

while True:
    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    state["userMessage"] = user_input

    state = workflow.invoke(state)

    if state["phase"] == "init":
        assistant_reply = state["deepResearch"]
    else:
        assistant_reply = state["userRelatedResearch"]

    print("\nAI Research Agent:\n", assistant_reply)

    state["chat_history"].append(f"User: {user_input}")
    state["chat_history"].append(f"AI Research Agent: {assistant_reply}")
