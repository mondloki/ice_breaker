import os

from dotenv import load_dotenv
import sys

current_dir = os.path.dirname(__file__) 

print("*"* 100)
print(current_dir)
# sys.path.append("..")


load_dotenv()

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor, 
    )
from langchain import hub
from tools.agent_tools import get_profile_url_tavily
from langchain_google_genai import ChatGoogleGenerativeAI

# model = "llama3.1:8b-instruct-q8_0"
def lookup(name: str) -> str:
    # llm =  ChatOllama(model=model, temperature=0)
    

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""
    
    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, 
                                   tools=tools_for_agent, 
                                   verbose=True, 
                                   handle_parsing_errors=True,
                                   return_intermediate_steps=False,
                                   early_stopping_method="generate"
                                   )

    result = agent_executor.invoke(
        input={"input" : prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = result["output"]

    return linkedin_profile_url


if __name__ == "__main__":
    linkedin_url = lookup(name="Lokesh siva kumar Mondreti")
    print("*"*100)
    print(linkedin_url)
    print("*"*100)