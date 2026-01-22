from traceback import print_tb
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

response = tavily_client.search(
    query="Artificial Intelligence and Recent Investment",
    search_depth="advanced",      
    include_raw_content=False,    
    # max_results=5
)

# documents = []
# combined_content = ""

# print(len(response["results"]))

# for r in response['results']:
    # combined_content += r['content']
    # print(r['content'])
    # print(json.dump(data=r, indent=2))
    # print("\n\n")

# print(combined_content)

# for r in response["results"]:
#     if r.get("content"):
#         documents.append(
#             f"{r['content']}"
#         )

# combined_content = "\n\n".join(documents)

# print(combined_content)
