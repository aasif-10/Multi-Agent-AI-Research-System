from agents import build_scrape_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic : str) -> dict:
    state ={}
    
    print("\n" + "=" * 60)
    print(f"🚀 STARTING RESEARCH ON: {topic}")
    print("=" * 60 + "\n")

    #search agent working
    print("🔍 [1/4] Running Search Agent...")
    search_agent = build_search_agent()

    search_result = search_agent.invoke({
        "messages" : [
            ("user",f"Find recent, reliable and detailed information about: {topic}. Please make sure you include the complete(full) URLs of the sources in your final response.")
        ]
    })

    raw_search = search_result['messages'][-1].content
    if isinstance(raw_search, list):
        state['search_result'] = "".join(item.get("text", "") for item in raw_search if item.get("type") == "text")
    else:
        state['search_result'] = raw_search
    
    print("-" * 60)
    print(">>> SEARCH RESULTS:")
    print("-" * 60)
    print(state['search_result'])
    print("-" * 60 + "\n")

    # scrape agent working
    print("🕷️ [2/4] Running Scrape Agent...")
    scrape_agent =  build_scrape_agent()
    scrape_result = scrape_agent.invoke({
        "messages" : [
            ("user",f"Based on the following search results about '{topic}', pick the most relevant URL and scrape it for deeper content.\n\nSearch Results:\n {state['search_result'][:800]}"
        )]
    })

    state['scrape_result'] = scrape_result['messages'][-1].content
    
    print("-" * 60)
    print(">>> SCRAPE RESULTS:")
    print("-" * 60)
    print(state['scrape_result'])
    print("-" * 60 + "\n")

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']}"
        f"DETAILED SCRAPED CONTENT: \n {state['scrape_result']}"
    )

    report = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    state['report'] = report
    
    print("📝 [3/4] Generating Report...")
    print("-" * 60)
    print(">>> DRAFT REPORT:")
    print("-" * 60)
    print(state['report'])
    print("-" * 60 + "\n")

    critic = critic_chain.invoke({
        "report" : state['report']
    })

    state["feedback"] = critic
    
    print("🧐 [4/4] Generating Critique...")
    print("-" * 60)
    print(">>> CRITIC FEEDBACK:")
    print("-" * 60)
    print(state['feedback'])
    print("-" * 60 + "\n")

    return state

    


if __name__ == "__main__":
    topic = input("\n Enter a research topic: ")

    run_research_pipeline(topic)