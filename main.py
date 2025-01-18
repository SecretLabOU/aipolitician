from swarms import Agent

agent = Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

agent.run("What is the current market trend for tech stocks?")
