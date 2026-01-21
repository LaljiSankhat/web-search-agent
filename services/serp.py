from serpapi import GoogleSearch
import json
from dotenv import load_dotenv
load_dotenv()
import os



params = {
    "q": "latest AI news",          # your search query
    "location": "Austin,Texas",     # location
    "api_key": os.getenv("SERP_API_KEY"),  # your SerpApi API key
}

search = GoogleSearch(params)
results = search.get_dict()          # parse to Python dict
print(json.dumps(results["organic_results"], indent=4))    # show organic search results

l = [news['link'] for news in results['organic_results']]

print(l)
