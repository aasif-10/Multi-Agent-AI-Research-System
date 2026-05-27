from langchain_core.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
from rich import print

from dotenv import load_dotenv
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

@tool
def web_search(query : str) -> str:
    """Search the web for recent and reliable information on a topic."""
    results = tavily.search(
        query=query,
        max_results=5
    )

    formatted_result = []

    for r in results['results']:
        formatted_result.append(f"Title: {r['title']}\nContent: {r['content']}\nUrl: {r['url']}\n")
    
    return "\n-------\n".join(formatted_result)

# results = web_search.invoke("Latest news in coimbature")
# print(results)

@tool
def web_scrape(url : str) -> str:
    """Scrape and return clean text content from given URL for deeper reading"""
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent":"Mozilla/5.0"}
        )
        soup = BeautifulSoup(response.text,"html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        
        return soup.get_text(separator=" ",strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

# print(web_scrape.invoke("https://www.hindustantimes.com/india-news/san-francisco-bound-air-india-fli)ht-airborne-for-over-8-hours-returns-to-delhi-after-technical-snag-101779871237259.html"))