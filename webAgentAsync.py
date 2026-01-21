from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from services.llms import map_llm, deep_llm
from services.prompts import deep_think_prompt, user_interest_prompt, map_prompt
from services.text_splitter import split_text_into_chunks
from langgraph.types import interrupt, Command
from services.async_tavily import async_tavily_client
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

load_dotenv()
from langfuse.langchain import CallbackHandler
from langfuse import observe

langfuse_handler = CallbackHandler()


# search = TavilySearch()

class AgentState(TypedDict):
    # Chat
    userMessage: str

    # Research
    topic: Optional[str]
    contents: str
    summarized_content: str
    deepResearch: str

    # Refinement
    userInterest: Optional[str]
    userRelatedResearch: Optional[str]

    satisfied: Optional[bool]


@observe(name="get-content")
async def get_contents(state: AgentState):
    topic_user = state["userMessage"]
    state["topic"] = topic_user

    response = await async_tavily_client.search(
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

    state["contents"] = combined_content
    return state


@observe(name="deep-think")
async def deep_think(state: AgentState):
    result = state['contents']
    state["summarized_content"] = ""


    chunks = split_text_into_chunks(result)

    chunk_summaries = []

    print(f"Total chunks to analyse {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i+1}...")
        chunk_response = await map_llm.ainvoke(map_prompt.format(chunk_content=chunk))
        chunk_summaries.append(chunk_response.content)

    # final_combined_content = "\n\n".join(chunk_summaries)
    state['summarized_content'] = "\n\n".join(chunk_summaries)

    print("Starting final Deep Think analysis...")

    # 4. Final Deep Think Call
    response = await deep_llm.ainvoke(
        deep_think_prompt.format_messages(
            topic=state['topic'],
            combined_content=state['summarized_content']
        )
    )

    print(response.content)

    state['deepResearch'] = response.content
    return state

@observe(name="Human-approval")
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


@observe(name="satisfaction")
def satisfaction(state: AgentState):
    decision = interrupt(
        {"question": "Are you satisfied?", "options": ["yes", "no"]}
    )
    return {"satisfied": decision == "yes"}


@observe(name="refine-research")
async def refined_research(state: AgentState):

    response = await deep_llm.ainvoke(
        user_interest_prompt.format_messages(
            topic=state['topic'],
            interest=state['userInterest'],
            combined_content=state['summarized_content']
        )
    )

    state['userRelatedResearch'] = response.content
    print(state['userRelatedResearch'])
    return state




# memory = MemorySaver()



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
        False: "refined_research", 
    }
)


config = {
    "configurable": {"thread_id": "conversation_1"},
    "callbacks": [langfuse_handler]
}

# config = {"configurable": {"thread_id": "conversation_1"}}
# workflow = graph.compile(
#     checkpointer=memory
# )

DB_URL = os.getenv("DB_URL")

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as memory:

        # run only one time to create database tables``
        # await memory.setup()

        workflow = graph.compile(
            checkpointer=memory
        )
        print("AI Research Agent: Hello! What topic would you like to research today?")


        while True:
            user_topic = input("\nEnter topic (or exit): ").strip()
            if user_topic.lower() in ["exit", "quit"]:
                break

            await workflow.ainvoke(
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
                snapshot = await workflow.aget_state(config)

                if not snapshot.interrupts:
                    break

                interrupt_data = snapshot.interrupts[0].value
                print("\n", interrupt_data["question"])
                print("Options:", interrupt_data["options"])

                user_input = input("> ").strip().lower()

                # If refinement needed, ask sub-topic
                if interrupt_data["question"].startswith("Do you want to research") and user_input == "yes":
                    sub_topic = input("\nEnter specific sub-topic: ").strip()
                    await workflow.ainvoke(
                        Command(resume=user_input, update={"userInterest": sub_topic, "userRelatedResearch": None}),
                        config
                    )
                else:
                    await workflow.ainvoke(Command(resume=user_input), config)

            print("\n Research cycle completed ...")



asyncio.run(main()) 