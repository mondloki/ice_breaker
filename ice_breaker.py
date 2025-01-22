from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import warnings
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import summary_parser
warnings.simplefilter("ignore") 
# model = "llama3.2:3b-instruct-fp16"

from langchain_google_genai import ChatGoogleGenerativeAI
def ice_break_with(name:str) -> str:
    print("**App started**")
    linkedin_url_free_text = linkedin_lookup_agent(name=name)
    linkedin_profile_url = linkedin_url_free_text
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url, mock=True)

    summary_template = """
    given the Linkedin information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them

    \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template,
                                             partial_variables={"format_instructions" : summary_parser.get_format_instructions()})

    # llm = ChatOllama(model=model)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    chain = summary_prompt_template | llm | summary_parser
    print("**llm call started**")
    result = chain.invoke(input={"information" : linkedin_data})
    print("$"*50)
    print(result)
    print("$"*50)
    print("**App run completed**")

if __name__ == "__main__":
    ice_break_with("Lokesh siva kumar Mondreti")
#     information = """
# Mark Elliot Zuckerberg (born May 14, 1984) is an American businessman who co-founded 
# the social media service Facebook and its parent company Meta Platforms, of which he is the chairman,
# chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits 
# regarding the creation and ownership of the website as well as issues such as user privacy.
# Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his 
# roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. 
# Zuckerberg took the company public in May 2012 with majority shares. 
# """
    

