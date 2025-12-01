# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.pretty import pprint
import mlflow
import dspy
import os

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("20_days_dspy")
mlflow.dspy.autolog()

# mlflow.dspy.autolog(
#     log_compiles=True,    # Track optimization process
#     log_evals=True,       # Track evaluation results
#     log_traces_from_compile=True  # Track program traces during optimization
# )

# 1. Configure the Language Model
#    Replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key.
#    You can also set it as an environment variable: export OPENAI_API_KEY='your_key_here'
#    If using a local LM with Ollama or SGLang, adjust api_base and api_key accordingly.

LLM_URL=os.getenv('LLM_URL', 'http://localhost:8080/v1')
API_KEY=os.getenv('API_KEY', 'fake')
LLM_MODEL=os.getenv('LLM_MODEL', 'openai/models/Llama-3.2-3B-Instruct-Q8_0.gguf')
MAX_TOKENS=os.getenv('MAX_TOKENS', 3000)
TEMPERATURE=os.getenv('TEMPERATURE', 0.2)
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY', 'fake')
URL = "https://mcp.tavily.com/mcp/?tavilyApiKey=" + TAVILY_API_KEY

dspy.enable_logging()

lm = dspy.LM(model=LLM_MODEL,
             api_base=LLM_URL,
             api_key=API_KEY,
             temperature=TEMPERATURE,
             model_type='chat',
             stream=False,
             cache=False)
dspy.configure(lm=lm)
dspy.settings.configure(track_usage=True)

server_params = StdioServerParameters(
    command="python",
    args=["mcp-proxy4.py"],
    env={"TAVILY_API_KEY": TAVILY_API_KEY, "URL": URL, "DEBUG_MODE": "False"},
)

colbertv2 = dspy.ColBERTv2(
    url="http://127.0.0.1:8081/api/search"
)
dspy.settings.configure(rm=colbertv2)

class RAGQA(dspy.Signature):
    """Answer questions based on the provided context."""

    context = dspy.InputField(desc="relevant passages from a knowledge base")
    question = dspy.InputField()
    answer = dspy.OutputField()

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)  # Retrieve top-k passages
        self.generate_answer = dspy.ChainOfThought(RAGQA)  # Generate answer using ChainOfThought

    def forward(self, question):
        # Perform retrieval
        context = self.retrieve(question).passages
        # Generate answer using the retrieved context
        prediction = self.generate_answer(context=context, question=question)
        return prediction

def search_developers(query: str) -> list[str]:
    """Search for answers based on the provided context"""
    rag_program = BasicRAG()
    prediction_rag = rag_program(question=query)
    return prediction_rag.answer

class PodLogsSummarize(dspy.Signature):
    """You are an expert OpenShift administrator. Your task is to analyze pod logs and summarize the error. Review the OpenShift logs for the pod 'java-app-build-run-bad-80uy3b-build-pod' in the 'demo-pipeline' namespace. If the logs indicate an error, create a summary message with the category and explanation of the error."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Summary message with the category and explanation of the error"
        )
    )

async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            dspy_tools.append(search_developers)
            print(f"Number of Tools: {len(dspy_tools)}")
            #pprint(dspy_tools)

            # Create the agent
            react = dspy.ReAct(PodLogsSummarize, tools=dspy_tools, max_iters=20)

            result = await react.acall(user_request=user_request)
            pprint(result)

            dspy.inspect_history()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run("""You are an expert OpenShift administrator. Your task is to analyze pod logs and summarize the error. Review the OpenShift logs for the container 'step-s2i-build' in the pod 'java-app-build-run-bad-80uy3b-build-pod' in the 'demo-pipeline' namespace. If the logs indicate an error, search for answers returning any advice found. Create a summary message with the error category, an explanation of the error and your reasoning. Create a Github issue using {"name":"create_issue","arguments":{"owner":"redhat-ai-services","repo":"etx-agentic-ai","title":"Issue with Etx pipeline","body":"<summary of the error>"}} DO NOT add any other parameters to the tool call."""))
