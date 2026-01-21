# from langgraph.checkpoint.postgres import PostgresSaver


# DB_URL = "postgres://postgres:postgres@localhost:5432/webAgentDemo"

# config = {
#     "configurable": {
#         "thread_id": "demo-thread"
#     }
# }

# with PostgresSaver.from_conn_string(DB_URL) as memory:
#     memory.setup()   
#     checkpoints = list(memory.list(config, limit=2))
#     print(checkpoints)


# import asyncio
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver



# DB_URL = "postgres://postgres:postgres@localhost:5432/webAgentDemoAsync"

# async def setup_db():
#     async with AsyncPostgresSaver.from_conn_string(DB_URL) as memory:
#         await memory.setup()

# asyncio.run(setup_db())


import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DB_URL = "postgres://postgres:postgres@localhost:5432/webAgentDemoAsync"

class State(TypedDict):
    count: int

async def add_one(state: State):
    return {"count": state["count"] + 1}



workflow = StateGraph(State)
workflow.add_node("add", add_one)
workflow.add_edge(START, "add")
workflow.add_edge("add", END)


config = {
    "configurable": {
        "thread_id": "demo-thread"
    }
}

async def main():
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as memory:
        # one-time (safe to keep commented after first run)
        # await memory.setup()

        graph = workflow.compile(checkpointer=memory)

        result = await graph.ainvoke({"count": 1}, config)
        print(result)

asyncio.run(main())
