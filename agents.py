from swarms import Agent

agent = Agent(
    agent_name="Trump",
    model_name="gpt-4o-mini",
    max_loops="auto",
    interactive=True,
    streaming_on=True,
)

agent.run("Who is Donald Trump")