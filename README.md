# Multi-Agent Orchestration Pattern

## Overview
# Multi Agent Chat Bot

I’m ReAct, your intelligent assistant designed to Reason and Take Actions effectively. 

To support you better, I work alongside three specialized subagents:

Docu-Talk: Your go-to expert for document-related queries.

Ask-Finance: Here to assist with all your finance-related questions.

General bot - Your friendly co-pilot

# An Orchestration Pattern

An Orchestration Pattern in an Agentic Framework refers to the structured coordination of agents and processes to execute a task or workflow effectively. Here an LLM with Reasoning capabilities can be used as the Orchestrator, often called a Master agent. The Master agent dynamically assigns (unlike Routing pattern, the subtasks aren't pre-defined) task to worker agents, responsible for coordinating and overseeing the collaborative tasks of worker agents, and later reduces the final output.
This pattern is well suited for building complex Multi-Agent System. For Example:
A user uploads a pdf that has both text and images embedded to a chatbot and ask to summarize. The Master agent decides to orchestrate multiple worker agents – one for analyzing text to text, and another one to analyze image to text. The final output is later synthesized (post-process) and presented to the user.



## Installation & Setup
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```


## Implementation Details
## Setup

First, let's install our required packages and set our API keys
<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>
## Create Tools

Each team will be composed of one or more agents each with one or more tools. Below, define all the tools to be used by your different teams.

We'll start with the Search team.


# OCI NL2SQL Tool

![nl2sql.png](attachment:47d3a64b-789d-4c1e-ba34-ad6895880d47.png)
**OCI RAG AGENT SERVICE**

The OCI RAG Agent Service is a pre-built service from Oracle cloud, that is designed to perform multi-modal augmented search against any pdf (with embedded tables and charts) or txt files.

## Helper Utilities

We are going to create a few utility functions to make it more concise when we want to:

1. Create a worker agent.
2. Create a supervisor for the sub-graph.

These will simplify the graph compositional code at the end for us so it's easier to see what's going on.
## Define Agent Teams

Now we can get to define our hierarchical teams. "Choose your player!"

### ReAct Team

The ReAct team will have a RAG agent and a NL2SQL "react_agent" as the two worker nodes. Let's create those, as well as the team supervisor.
Now that we've created the necessary components, defining their interactions is easy. Add the nodes to the team graph, and define the edges, which determine the transition criteria.
We can give this team work directly. Try it out below.

## Code
```python
from IPython.display import Image
Image("images/multi-agent-react.png", width=800, height=600)

```
```python
from IPython.display import Image
Image("images/react.png", width=800, height=600)
```
```python
from IPython.display import Image
Image("images/multi-agent.png", width=800, height=600)

```
```python
%%capture --no-stderr
%pip install -U langgraph langchain_community langchain_anthropic langchain_experimental
```
```python
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain.prompts import PromptTemplate  # For creating prompts 


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

```
```python
import getpass
import os


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("TAVILY_API_KEY")
```
```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

@tool
def nl2sql_agent_service(textinput) -> str:
    """
    NL2SQL agent to answer structured query from a database"""
    nl2sql_tavily_tool = TavilySearchResults(max_results=3)
    response = nl2sql_tavily_tool.invoke(textinput)
    return response
```
```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=3)
```
```python
from langchain_core.tools import tool
from typing import Annotated, List
from langchain_community.document_loaders import WebBaseLoader


@tool
def scrape_webpages(query: str, urls: List[str]) -> str:
    """
    Scrape web pages for relevant information based on a user query.

    **Usage Guidelines:**
    - Use this tool when a user requests specific information from online sources.
    - This tool takes both a **query** and a list of **URLs** to retrieve and summarize content.
    - The returned information is **filtered** to match the query.

    **Example Usage:**
    ```python
    scrape_webpages("Latest AI advancements", ["https://ai-news.com/latest"])
    ```
    """

    if not urls:
        return "Error: No URLs provided for scraping."

    # Load web pages
    loader = WebBaseLoader(urls)
    docs = loader.load()

    # Extract relevant text based on the query
    extracted_content = []
    for doc in docs:
        title = doc.metadata.get("title", "Untitled Document")
        content = doc.page_content

        # Simple filtering: Only keep paragraphs that contain words from the query
        relevant_sentences = [
            sentence for sentence in content.split(". ")
            if any(word.lower() in sentence.lower() for word in query.split())
        ]
        
        if relevant_sentences:
            extracted_content.append(
                f'<Document name="{title}">\n' + ". ".join(relevant_sentences) + "\n</Document>"
            )

    # Return formatted content or an error message
    if extracted_content:
        return "\n\n".join(extracted_content)
    else:
        return f"No relevant content found for query: {query}"

```
```python
from IPython.display import Image
Image("images/rag.png", width=1000, height=800)
```
```python
import oci
from langchain_core.tools import tool

# Set up the OCI GenAI Agents endpoint configuration
agent_ep_id = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqaye4c3tcmg6rehfougltaikbo763b3v4tcyraac6snvfa"
config = oci.config.from_file(profile_name="DEFAULT") # Update this with your own profile name
service_ep = "https://agent-runtime.generativeai.us-chicago-1.oci.oraclecloud.com" # Update this with the appropriate endpoint for your region, a list of valid endpoints can be found here - https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-agents-client/20240531/

# Response Generator
@tool
def rag_agent_service(textinput) -> str:
    """TAny questions about US Tax, use the rag_agent tool. Don't use Search or Web_Search for questions related to Tax"""
    # Initialize service client with default config file
    generative_ai_agent_runtime_client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(config,service_endpoint=service_ep)
 
    # Create Session
    create_session_response = generative_ai_agent_runtime_client.create_session(
        create_session_details=oci.generative_ai_agent_runtime.models.CreateSessionDetails(
            display_name="USER_Session",
            description="User Session"),
        agent_endpoint_id=agent_ep_id)
 
    sess_id = create_session_response.data.id
 
    response = generative_ai_agent_runtime_client.chat(
        agent_endpoint_id=agent_ep_id,
        chat_details=oci.generative_ai_agent_runtime.models.ChatDetails(
            user_message=textinput,
            session_id=sess_id))
 
    #print(str(response.data))
    response = response.data.message.content.text
    return response
 
```
```python
import json
import re
from typing import List, Optional, Literal, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


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
    Supervisor function that routes user queries to the appropriate LangGraph sub-agent.
    It ensures that:
      - US Tax Law queries go to RAG
      - Other knowledge-based questions go to Web Search
      - Specific structured data queries go to Search
    """

    options = ["FINISH"] + members
    system_prompt = (
        "You are an expert AI supervisor managing a team of sub-agents."
        "\nYour task is to route user questions to the correct sub-agent based on the query topic."
        "\nDonot route to the same tool once done calling. If no answer received, route to the next available tool or return: {\"next\": \"FINISH\"}."
        "\nThe available agents are:"
        "\n- RAG: Handles US Tax Law-related questions (IRS, deductions, tax returns, laws, regulations)."
        "\n- Web_Search: Handles general knowledge and internet-related questions."
        "\n- search: Handles structured database queries."
        "\n\n**Instructions:**"
        "\n1️⃣ If the question is about US Tax Laws, IRS, deductions, or legal tax rules, route to: **RAG**."
        "\n2️⃣ If the question requires general internet-based answers, route to: **Web_Search**. If you can't find an answer, route to: **search**"
        "\n3️⃣ If the question requires converting a structured prompt to a PLSQL, route to: **nl2sql**."
        "\n4️⃣If the question requires data retrieval, route to: **search**."
        "\n5️⃣ If the conversation is complete, return: {\"next\": \"FINISH\"}."
        "\n\n**Examples:**"
        "\nUser: 'what are the US tax implicaitons for 2025'"
        "\nResponse: {\"next\": \"RAG\"}"
        "\n\nUser: 'Find me the latest news on AI regulations.'"
        "\nResponse: {\"next\": \"Web_Search\"}"
        "\n\nUser: 'What is the latest revenue report for Tesla?'"
        "\nResponse: {\"next\": \"search\"}"
        "\n\nUser: 'What are the top 10 outstanding invoices?'"
        "\nResponse: {\"next\": \"nl2sql\"}"
        
    )

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

```
```python
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

rag_agent = create_react_agent(llm_oci, tools=[rag_agent_service])


def rag_node(state: State) -> Command[Literal["supervisor"]]:
    result = rag_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="RAG")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


search_agent = create_react_agent(llm_oci, tools=[tavily_tool])


def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

web_scraper_agent = create_react_agent(llm_oci, tools=[scrape_webpages])


def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="Web_Search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

nl2sql_agent = create_react_agent(llm_oci, tools=[nl2sql_agent_service])


def nl2sql_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="nl2sql")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

research_supervisor_node = make_supervisor_node(llm_oci, ["RAG", "Web_Search", "search", "nl2sql"])
```
```python
research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("RAG", rag_node)
research_builder.add_node("search", search_node)
research_builder.add_node("Web_Search", web_scraper_node)
research_builder.add_node("nl2sql", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()
```
```python
from IPython.display import Image, display

display(Image(research_graph.get_graph().draw_mermaid_png()))
```
```python
for s in research_graph.stream(
    {"messages": [("user", "what are the tax implications for corporations in FY 2025")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
```
```python
for s in research_graph.stream(
    {"messages": [("user", "What is the latest revenue report for Tesla?")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")

```
```python
for s in research_graph.stream(
    {"messages": [("user", "Find recent research papers on quantum computing. , [https://arxiv.org/list/quant-ph/recent]")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
```
```python
for s in research_graph.stream(
    {"messages": [("user", "What are the top 10 outstanding invoices?")]},
    {"recursion_limit": 100},
):
    print(s)
    print("---")
```
```python

```

## Usage
Run the following command to execute the notebook:
```bash
jupyter notebook multi-agent-orchestration_pattern.ipynb
```
# agents
