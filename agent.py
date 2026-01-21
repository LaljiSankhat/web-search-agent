from typing import TypedDict, List
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
    topic: str
    sources: List[str]
    contents: List[str]
    deepResearch: str
    userInterest: str
    userRelatedResearch: str
    summarized_content: str



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


# def get_contents(state: AgentState):
#     contents = []
#     headers = {
#         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }
#     for url in state['sources']:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status() 

#         soup = BeautifulSoup(response.content, 'html.parser')
#         main_content_container = soup.find('div', class_='article--viewer_content')

#         if main_content_container:
#             paragraphs = main_content_container.find_all('p')
#             s = ""
#             for para in paragraphs:
#                 s += para.text.strip()
#             contents.append(s)
#         else:
#             print("Could not find the main content container. Check the website's HTML structure.")
#     state['contents'] = contents
#     return state


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

graph.add_node("get_sources", get_sources)
graph.add_node("get_contents", get_contents)
graph.add_node("deep_think", deep_think)
graph.add_node("refined_research", refined_research)


# adding edges to graph
graph.add_edge(START, "get_sources")
graph.add_edge("get_sources", "get_contents")
graph.add_edge("get_contents", "deep_think")
graph.add_edge("deep_think", "refined_research")
graph.add_edge("refined_research", END)



workflow = graph.compile()



initial_state = {
    "topic": "Artificial Intelligence",
    "userInterest": "Deep Learning"
}


final_state = workflow.invoke(initial_state)

print(final_state)

