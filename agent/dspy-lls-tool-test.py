import sys
import dspy
import logging
import os
import asyncio
import re

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.transports import SSETransport # deprecated now https://mcp-framework.com/docs/Transports/sse
from fastmcp.client.transports import StreamableHttpTransport
from llama_stack_client import LlamaStackClient

from rich import print

load_dotenv()
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL") # LLAMA_STACK_URL=http://localhost:8321, LLAMA_STACK_URL=http://llamastack-with-config-service.llama-stack.svc.cluster.local:8321


root = logging.getLogger()
root.setLevel(logging.WARN)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


class CustomerAssistantService(dspy.Signature):
    """You are a helpful customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
        )
    )


async def log_handler(message: LogMessage):
    level = message.level.upper()
    logger = message.logger or "server"
    data = message.data
    print(f"[{level}] {logger}: {data}")


def lls_get_tools():
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    tool_group = lls_client.toolgroups.list()
    mcp_tools = []
    for tool in tool_group:
        #print(f">>> Tool: {tool.provider_id}")
        if (tool.provider_id == "model-context-protocol"
            or tool.provider_id == "tavily-search"):
            #or tool.provider_id == "rag-runtime"):
            mcp_tools.append(tool) 

    tool_list = {}
    for tool in mcp_tools:
        if (tool.provider_id == "model-context-protocol"):
            tool_list[tool.identifier] = tool.mcp_endpoint.uri
        elif (tool.provider_id == "tavily-search"):
            tool_list[tool.identifier] = 'websearch'
    return tool_list

async def builtin_websearch(query: str):
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    response = lls_client.tool_runtime.invoke_tool(
        tool_name="web_search", kwargs={"query": query}
    )
    print(response.json())
    return response

async def convert_tools_dspy() -> list[dspy.adapters.types.tool.Tool]:
    tools_list = lls_get_tools()
    dspy_tools = []

    for tool in tools_list:
        if tools_list[tool] == 'websearch':
            dspy_tools.append(dspy.Tool(builtin_websearch))
            continue

        print(tools_list[tool])
        match = re.search(r"/sse", tools_list[tool])
        if match:
            mcp_client = Client(
                SSETransport(url=tools_list[tool]), log_handler=log_handler
            )
        else:
            mcp_client = Client(
                StreamableHttpTransport(url=tools_list[tool]), log_handler=log_handler
            )
        await mcp_client._connect()
        print(
            f" Connected:{mcp_client.is_connected()} Init result:{mcp_client.initialize_result}\n"
        )
        provided_tools = await mcp_client.list_tools()

        for method in provided_tools:
            dspy_tools.append(dspy.Tool.from_mcp_tool(mcp_client.session, method))
    return dspy_tools


async def dspy_mcp(user_request: str):
    dspy_tools = await convert_tools_dspy()
    print(dspy_tools)
    react = dspy.ReAct(CustomerAssistantService, tools=dspy_tools)
    result = await react.acall(user_request=user_request)
    #print(result)
    dspy.inspect_history(n=50)


def lls_get_inference_server():
    lls_client = LlamaStackClient(base_url=LLAMA_STACK_URL)
    model_list = lls_client.models.list()
    print(model_list)
    llm = dspy.LM(
        "openai/" + model_list[2].identifier, # hack for llama-4-scout-17b-16e-w4a16
        api_base=LLAMA_STACK_URL + "/v1/openai/v1",
        model_type="chat",
    )
    return llm


if __name__ == "__main__":
    dspy.enable_logging()
    llm = lls_get_inference_server()

    dspy.configure(lm=llm)
    asyncio.run(
        dspy_mcp(
            "help me find the recipe for spicy mexican tacos? just do a basic search"
        )
    )
