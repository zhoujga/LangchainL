# pip install -qU langchain "langchain[openai]"
import loadenv

from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime

load_dotenv()

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
agent = create_agent(
    model="deepseek:deepseek-v4-flash",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",

)

result = agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]},
    stream_mode=["values","custom"]

)

for trunk in result:
    print(trunk)

print(result)

#print(result["messages"][-1].content_blocks)