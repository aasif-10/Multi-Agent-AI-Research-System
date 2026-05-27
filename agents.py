from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from tools import web_search,web_scrape

from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(
    model = "mistral-small-2506",
    temperature = 0
)

# 1st agent
def build_search_agent():
    return create_agent(
        model = model,
        tools = [web_search]
    )

# 2nd agent
def build_scrape_agent():
    return create_agent(
        model = model,
        tools = [web_scrape]  
    )

# writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.
     
Topic: {topic}

Research Gathered:
{research}
     
Structure the report as:
-Introduction
-Key Findings (minimum 3 well-explained points)
-Conclusion
-Sources (list all URLs found in the research)
     
Be detailed, factual and professional."""),
])

output_parser = StrOutputParser()

writer_chain = writer_prompt | model | output_parser

# critic chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | model | output_parser