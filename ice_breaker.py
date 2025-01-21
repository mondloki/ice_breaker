from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import warnings
from third_parties.linkedin import scrape_linkedin_profile
warnings.simplefilter("ignore") 



if __name__ == "__main__":
    print("**App started**")
#     information = """
# Mark Elliot Zuckerberg (born May 14, 1984) is an American businessman who co-founded 
# the social media service Facebook and its parent company Meta Platforms, of which he is the chairman,
# chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits 
# regarding the creation and ownership of the website as well as issues such as user privacy.
# Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his 
# roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. 
# Zuckerberg took the company public in May 2012 with majority shares. 
# """
    summary_template = """
    given the information {information} of a person, I want to you to create:
    1. A short summary
    2. Interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)

    llm = ChatOllama(model="llama3.2")

    chain = summary_prompt_template | llm | StrOutputParser()
    print("**llm call started**")
    linkedin_json_data = scrape_linkedin_profile(linkedin_profile_url="https://www.linkedin.com/in/eden-marco/",mock=True)
    result = chain.invoke(input={"information" : linkedin_json_data})
    print(result)
    print("**App run completed**")

