from graphviz import Digraph
import os

output_dir = "Diagrams/Output"
os.makedirs(output_dir, exist_ok=True)

# Create a Data Flow Diagram (DFD) for real-time debate functionality
def create_political_debate_dfd():
    dfd = Digraph("Political Debate Data Flow Diagram", filename=os.path.join(output_dir, "political_debate_dfd"), format="png", engine="dot")
    dfd.attr(rankdir="TB", splines="polyline")

    # Nodes
    dfd.node("Initialize", "Initialize Agent with Knowledge", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    dfd.node("Receive", "Receive Input from User", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    dfd.node("Fetch", "Fetch Data (API, Memory)", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    dfd.node("Debate", "Real-Time Debate Among Agents", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    dfd.node("Update", "Update Knowledge Base", shape="box", style="rounded,filled", color="black", fillcolor="gray92")

    # Edges
    dfd.edge("Initialize", "Receive")
    dfd.edge("Receive", "Fetch")
    dfd.edge("Fetch", "Debate")
    dfd.edge("Debate", "Update")

    # Render Diagram
    dfd.render(view=True)

# Create a Swarm Architecture diagram reflecting real-time debate functionality
def create_political_debate_swarm_architecture():
    swarm = Digraph("Political Debate Swarm Architecture", filename=os.path.join(output_dir, "political_debate_swarm_architecture"), format="png", engine="dot")
    swarm.attr(rankdir="TB", splines="polyline")

    # Nodes
    swarm.node("Swarm", "Swarm (Group Chat)", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("UserQuery", "User Query Input", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("Agent1", "Agent 1 (Politician A)", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("Agent2", "Agent 2 (Politician B)", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("AgentN", "Agent N (Politician N)", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("Debate", "Real-Time Debate", shape="box", style="rounded,filled", color="black", fillcolor="gray92")
    swarm.node("Output", "Evolving Debate Output", shape="box", style="rounded,filled", color="black", fillcolor="gray92")

    # Edges
    swarm.edge("UserQuery", "Swarm")
    swarm.edge("Swarm", "Agent1")
    swarm.edge("Swarm", "Agent2")
    swarm.edge("Swarm", "AgentN")
    swarm.edge("Agent1", "Debate")
    swarm.edge("Agent2", "Debate")
    swarm.edge("AgentN", "Debate")
    swarm.edge("Debate", "Output")

    # Render Diagram
    swarm.render(view=True)

# Generate updated diagrams
if __name__ == "__main__":
    create_political_debate_dfd()
    create_political_debate_swarm_architecture()
