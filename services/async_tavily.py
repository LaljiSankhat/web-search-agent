import asyncio
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

async_tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# async def fetch_and_gather(topic: str):
    
#     response = await async_tavily_client.search(topic)
#     documents = []

#     for r in response["results"]:
#         if r.get("content"):
#             documents.append(
#                 f"{r['content']}"
#             )

#     combined_content = "\n\n".join(documents)

#     print(combined_content[:])


# asyncio.run(fetch_and_gather("Artificial Intelligence and machine learning"))