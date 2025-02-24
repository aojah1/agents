import oci
import os
import getpass
import re
import json
import requests

# For OCI GenAI Service
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI 

 # For creating prompts 
from langchain.prompts import PromptTemplate 
from langchain_core.messages import HumanMessage, SystemMessage

# code helper class

from operator import itemgetter
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing import List, Optional, Literal, TypedDict

# Lang chain helper class
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

# langgraph helper class
from langgraph.graph import START, MessagesState, StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# NL2SQL Demo
from langchain_community.utilities import SQLDatabase 
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import Annotated, List
from langchain_community.document_loaders import WebBaseLoader

# Web Scrapper
from bs4 import BeautifulSoup



"""
Multi Agent Chat Bot
I’m ReAct, your intelligent assistant designed to Reason and Take Actions effectively.

To support you better, I work alongside three specialized subagents:

Docu-Talk: Your go-to expert for document-related queries.

Ask-Finance: Here to assist with all your finance-related questions.

General bot - Your friendly co-pilot

"""

# Global variables for LLM and tools
llm_oci = None
tavily_tool = None

# Define SQLite database file
DATABASE_URL = "sqlite:///Chinook.db"
db = SQLDatabase.from_uri(DATABASE_URL) 

# Set up the OCI GenAI Agents endpoint configuration
agent_ep_id = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqaye4c3tcmg6rehfougltaikbo763b3v4tcyraac6snvfa"
config = oci.config.from_file(profile_name="DEFAULT") # Update this with your own profile name
service_ep = "https://agent-runtime.generativeai.us-chicago-1.oci.oraclecloud.com" 
generative_ai_agent_runtime_client = None
session_id = None

###### Setup  - START - #########
def initialize_llm():
    try:
        # Set your OCI credentials 
        compartment_id = "ocid1.compartment.oc1..aaaaaaaau6esoygdsqxfz6iv3u7ghvosfskyvd6kroucemvyr5wzzjcw6aaa" 
        service_endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com" 
        #model_id = "meta.llama-3.3-70b-instruct"
        model_id = "cohere.command-r-plus-08-2024"
        # Create an OCI Cohere LLM instance 
        llm_oci = ChatOCIGenAI(
            model_id= model_id,  # Specify the model you want to use 
            service_endpoint=service_endpoint,
            compartment_id=compartment_id,
            model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 1000}
        ) 
        return llm_oci
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise
        
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")



def initialize_oci_genai_agent_service():
    """Initialize OCI GenAI Agent Service and create a session"""
    
    

    # Initialize service client with default config file
    generative_ai_agent_runtime_client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(config,service_endpoint=service_ep)

    # Create Session
    create_session_response = generative_ai_agent_runtime_client.create_session(
        create_session_details=oci.generative_ai_agent_runtime.models.CreateSessionDetails(
            display_name="USER_Session",
            description="User Session"),
        agent_endpoint_id=agent_ep_id)

    sess_id = create_session_response.data.id
    
    print("generative_ai_agent_runtime_client :" + str(generative_ai_agent_runtime_client))
    print(sess_id)
    
    
    return generative_ai_agent_runtime_client, sess_id  # Return both client and session ID
    
###### Setup  - END - #########   


###### Create Tools #########

"""
Each team will be composed of one or more agents each with one or more tools. Below, define all the tools to be used by your different teams.
"""


### Clean SQL Queries

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.

    Args:
        text (str): Raw SQL query text that may contain code blocks, tags, and backticks

    Returns:
        str: Cleaned SQL query
    """
    # Step 1: Remove code block syntax and any SQL-related tags
    # This handles variations like ```sql, ```SQL, ```SQLQuery, etc.
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)

    # Step 2: Handle "SQLQuery:" prefix and similar variations
    # This will match patterns like "SQLQuery:", "SQL Query:", "MySQL:", etc.
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

    # Step 3: Extract the first SQL statement if there's random text after it
    # Look for a complete SQL statement ending with semicolon
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # Step 4: Remove backticks around identifiers
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # Step 5: Normalize whitespace
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Step 6: Preserve newlines for main SQL keywords to maintain readability
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
               'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
               'OUTER JOIN', 'UNION', 'VALUES', 'INSERT', 'UPDATE', 'DELETE']

    # Case-insensitive replacement for keywords
    pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
    text = re.sub(f'({pattern})', r'\n\1', text, flags=re.IGNORECASE)

    # Step 7: Final cleanup
    # Remove leading/trailing whitespace and extra newlines
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text

### Create the NL2SQL Tool

class SQLToolSchema(BaseModel):
    question: str

@tool(args_schema=SQLToolSchema)
def nl2sql_agent_service(question):
    """Tool to Generate and Execute SQL Query to answer User Questions related to chinook DB"""
    print("INSIDE NL2SQL TOOL")
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm_oci, db)

    chain = (
       RunnablePassthrough.assign(query=write_query | RunnableLambda(clean_sql_query)).assign(
           result=itemgetter("query") | execute_query
       )
    )
    
    preamble = "Tool to Generate and Execute SQL Query to answer User Questions related to chinook DB"
    prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=preamble),  # System message
    ("human", "{question}")])

    
    # Extracting the content
    response = chain.invoke({"question": question})
    
    #modified_resp = HumanMessage(content= response['result'])
    #print(modified_resp)
    
    return response


### Search Tool
tavily_tool = TavilySearchResults(max_results=3)

### Web Scrapper Tool

@tool
def scrape_webpages(query: str, urls: List[str]) -> str:
    """
    **Improved Web Scraper Tool**
    
    This tool scrapes ONLY the provided URLs for relevant information related to the query.

    **Enhancements:**
    - ✅ **Ensures URLs are provided & valid**
    - ✅ **Uses BeautifulSoup for better content extraction**
    - ✅ **Filters & ranks content based on semantic relevance**
    - ✅ **Summarizes results to improve readability**

    **Example Usage:**
    ```python
    scrape_webpages("Latest AI advancements", ["https://ai-news.com/latest"])
    ```
    """

    if not urls:
        return "Error: No URLs provided for scraping."

    extracted_content = []

    for url in urls:
        try:
            # Load webpage content
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Ensure the request was successful
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract relevant text from paragraphs
            paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
            full_text = "\n".join(paragraphs)

            # Improve text cleaning (remove excessive whitespace, JavaScript remnants, etc.)
            cleaned_text = re.sub(r"\s+", " ", full_text)

            # Extract only relevant sentences
            relevant_sentences = [
                sentence for sentence in cleaned_text.split(". ")
                if any(word.lower() in sentence.lower() for word in query.split())
            ]

            if relevant_sentences:
                extracted_content.append(f"📄 **[{url}]**\n" + ". ".join(relevant_sentences))

        except requests.exceptions.RequestException as e:
            extracted_content.append(f"❌ Error fetching {url}: {str(e)}")

    # Return formatted content or an error message
    if extracted_content:
        return "\n\n".join(extracted_content)
    else:
        return f"⚠️ No relevant content found for query: {query}"
    

### OCI RAG AGENT SERVICE



### RAG Agent Tool

@tool
def rag_agent_service(textinput) -> str:
    """Any questions about US Tax, use the rag_agent tool. Don't use search or Web_Scrapper for questions related to Tax"""
    generative_ai_agent_runtime_client, session_id = initialize_oci_genai_agent_service()
    response = generative_ai_agent_runtime_client.chat(
        agent_endpoint_id=agent_ep_id,
        chat_details=oci.generative_ai_agent_runtime.models.ChatDetails(
            user_message=textinput,
            session_id=session_id))
 
    #print(str(response.data))
    response = response.data.message.content.text
    return response

### Helper Utilities
"""
We are going to create a few utility functions to make it more concise when we want to:

Create a worker agent.
Create a supervisor for the sub-graph.
These will simplify the graph compositional code at the end for us so it's easier to see what's going on.
"""

class State(MessagesState):
    next: str


def extract_json(text: str) -> dict:
    """
    Extracts a valid JSON object from the text using regex.
    If multiple JSON objects exist, it picks the first one.
    """
    json_pattern = re.findall(r'\{.*?\}', text, re.DOTALL)
    
    for json_str in json_pattern:
        try:
            parsed_json = json.loads(json_str)

            # Handle nested JSON cases
            if isinstance(parsed_json, dict):
                if "supervisor" in parsed_json and "next" in parsed_json["supervisor"]:
                    return {"next": parsed_json["supervisor"]["next"]}
                if "next" in parsed_json:
                    return parsed_json
        except json.JSONDecodeError:
            continue  # Try the next match if this one fails
    
    return {"next": "FINISH"}  # Default if no valid JSON is found


def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    """
    Supervisor function responsible for routing user queries to the appropriate LangGraph sub-agent.
    It ensures that:
      - US Tax Law queries are handled by **RAG**
      - General knowledge-based queries are handled by **Web_Scrapper**
      - Internet-specific queries are handled by **search**
      - Natural language to SQL queries are handled by **nl2sql**
    """

    options = ["FINISH"] + members
    
    system_prompt = """
    Supervisor function responsible for routing user queries to the appropriate LangGraph sub-agent.
    It ensures that:
      - US Tax Law queries are handled by **RAG**
      - Ability to scrape a webpage when an URL is provided and extract information **Web_Scrapper**
      - Internet-specific queries are handled by **search**
      - Natural language to SQL queries are handled by **nl2sql**

    Your role is to intelligently route queries to the correct sub-agent while ensuring efficiency. 
    Avoid redundant tool calls, and if a tool fails, escalate to the next available option.
    If no relevant tool is found or the conversation is complete, return: {"next": "FINISH"}.

    ### **Available Agents & Responsibilities**
    - **RAG** → Handles **US Tax Law-related questions** (IRS regulations, tax deductions, legal compliance).
    - **Web_Scrapper** → Handles **general knowledge questions** that require web scraping.
    - **search** → Handles **real-time internet-based queries** (news, weather, current events).
    - **nl2sql** → Handles **natural language to SQL translation queries** for database access.

    ### **Routing Rules**
    1️⃣ **Tax-related queries** (IRS rules, deductions, legal tax implications) → Route to **RAG**.  
    2️⃣ **Scrape a webpage when an URL is provided and extract information → Route to **Web_Scrapper**.  
    3️⃣ **Real-time, internet-based questions** (live data like weather, news) → Route to **search**.  
    4️⃣ **Database queries in natural language** → Route to **nl2sql**.  
    5️⃣ **Do not call the same tool twice in succession** unless needed. If a tool fails, escalate to another tool if applicable.  
    6️⃣ **If no relevant tool is found, or conversation is complete, return:** {"next": "FINISH"}.

    ### **Examples for Better Routing**
    ❌ **Avoid vague or incorrect routing decisions. Follow these examples:**  

    **Example 1: US Tax Law Query**  
    **User:** What are the tax implications of capital gains in 2025?  
    **Response:** {"next": "RAG"}  

    **Example 2: Scrape a webpage*  
    **User:** Find recent research papers on quantum computing. , [https://arxiv.org/list/quant-ph/recent].  
    **Response:** {"next": "Web_Scrapper"}  

    **Example 3: Real-Time Query (Internet-based)**  
    **User:** What is the temperature of Louisville KY today?  
    **Response:** {"next": "search"}  

    **Example 4: NL to SQL Conversion Query**  
    **User:** Show me all employees earning more than $50,000.  
    **Response:** {"next": "nl2sql"}  

    **Example 5: Conversation is Complete**  
    **User:** Thanks, that’s all I needed.  
    **Response:** {"next": "FINISH"}  
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers are needed, route to FINISH."""
        next: Literal[tuple(options)]  # Corrected Literal usage

    def supervisor_node(state: State) -> Command[Literal[tuple(members) + ("__end__",)]]:
        """An LLM-based router for LangGraph-based sub-agents."""
        
        messages = [HumanMessage(content=system_prompt)] + state["messages"]
        
        # Get LLM response
        response_text = llm.invoke(messages).content  # Ensure we extract .content
        #print(response_text)
        response = extract_json(response_text)  # Use robust JSON extraction
        goto = response.get("next", "FINISH")  # Default to FINISH if missing
        
        # Ensure valid routing
        if goto not in members:
            print(f"⚠️ route received: {goto}, defaulting to FINISH")
            goto = "FINISH"

        if goto == "FINISH":
            goto = END

        print(f"📌 Routing user to: {goto}")  # Debugging log
        return Command(goto=goto, update={"next": goto})

    return supervisor_node


### Define Agent Teams

"""
ReAct Team
The ReAct team will have a RAG agent and a NL2SQL "react_agent" as the two worker nodes. 
Let's create those, as well as the team supervisor.
"""

def rag_node(state: State) -> Command[Literal["supervisor"]]:
    result = create_react_agent(llm_oci, tools=[rag_agent_service]).invoke(state)
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RAG")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )
  
def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = create_react_agent(llm_oci, tools=[tavily_tool]).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    result = create_react_agent(llm_oci, tools=[scrape_webpages]).invoke(state)
    print("result")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="Web_Scrapper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

def nl2sql_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = create_react_agent(llm_oci, tools=[nl2sql_agent_service]).invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="nl2sql")
            ]
        },
        goto="supervisor",
    )

def test_react(input_data) -> str:
    responses = []  # List to store all responses

    for s in research_graph.stream(
        {"messages": [("user", input_data)]},
        {"recursion_limit": 100}
    ):
        responses.append(str(s))  # Collect each response

    return "\n\n".join(responses)  # Join responses with spacing for readability

# Extract content from response dictionary
def print_message(response):
    print_string = ""
    if isinstance(response, dict):
        for agent, data in response.items():
            if "messages" in data and data["messages"]:
                print_string = data["messages"][0].content
                break
    else:
        print_string = response
    return print_string

if __name__ == "__main__":
    try:
        # Initialize components
        llm_oci = initialize_llm()
        #generative_ai_agent_runtime_client, session_id = initialize_oci_genai_agent_service()
        _set_if_undefined("TAVILY_API_KEY")
        
        tavily_tool = TavilySearchResults(max_results=3)
        
        research_supervisor_node = make_supervisor_node(llm_oci, ["RAG", "Web_Scrapper", "search", "nl2sql"])
        
        research_builder = StateGraph(State)
        research_builder.add_node("supervisor", research_supervisor_node)
        research_builder.add_node("RAG", rag_node)
        research_builder.add_node("search", search_node)
        research_builder.add_node("Web_Scrapper", web_scraper_node)
        research_builder.add_node("nl2sql", nl2sql_node)

        research_builder.add_edge(START, "supervisor")
        research_graph = research_builder.compile()
        
        
        
        search_input = "What is the temperature of Louisville KY today?"
        print("Response : ")
        print(print_message(test_react(search_input)))
        
        #rag_input = "what are the tax implications for corporations in FY 2025"
        #test_react(rag_input)
        
        #web_input = "Find recent research papers on quantum computing. , [https://arxiv.org/list/quant-ph/recent]"
        #test_react(web_input)
        
        #nl2sql_input = "show me all the employees"
        #test_react(nl2sql_input)
        
        

    except Exception as e:
        print(f"Fatal error: {e}")