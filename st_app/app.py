import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.tools.asknews import AskNewsSearch
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain_core.tools import StructuredTool
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_openai import AzureOpenAIEmbeddings
import requests

load_dotenv()

GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
WOLFRAM_ALPHA_ID = os.environ.get("WOLFRAM_ALPHA_APPID")

# google search tool
search = GoogleSearchAPIWrapper(
    k=2, google_api_key=GOOGLE_CSE_API_KEY, google_cse_id=GOOGLE_CSE_ID
)
tool = Tool(
    name="google_search",
    description="useful for when you need to answer questions about current events. You should ask targeted questions",
    func=search.run,
)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# integer multiplication tool
tool2 = StructuredTool.from_function(func=multiply, coroutine=amultiply)

# wolfram alpha calculator tool
# wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_ID)
# tool2 = Tool(
#     name="wolfram_alpha",
#     description="useful for when you need to do calculations. You should ask targeted questions",
#     func=wolfram.run
# )

# Youtube search tool
tool3 = YouTubeSearchTool()


def random_joke():
    """get a random joke"""
    response = requests.get("https://official-joke-api.appspot.com/random_joke")
    return response.text


async def arandom_joke():
    """get a random joke"""
    response = requests.get("https://official-joke-api.appspot.com/random_joke")
    return response.text


# tool that generates random jokes
tool4 = StructuredTool.from_function(func=random_joke, coroutine=arandom_joke)
tools = [tool, tool2, tool3, tool4]

# prompts
system = """You are a helpful assistant. Answer the questions based on the history of the conversation
and use tools to get the most recent information when necessary."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


# get the agent
def get_agent() -> AgentExecutor:
    llm = AzureChatOpenAI(
        temperature=0.0,
        model="gpt-4o-mini",
        api_version="2024-10-21",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor = get_agent()

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # LLM and tool usage "thoughts"
        st_callback = StreamlitCallbackHandler(st.container())
        # agent calling
        response = agent_executor.invoke(
            {
                "chat_history": st.session_state.chat_history,
                "user_question": user_query,
            },
            {"callbacks": [st_callback]},
        )
        st.write(response["output"])
        # response = st.write_stream(get_response(user_query, st.session_state.chat_history,agent_executor))

    st.session_state.chat_history.append(AIMessage(content=response["output"]))
