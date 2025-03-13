from setuptools import setup, find_packages

setup(
    name="ai-politician-langgraph-studio",
    version="0.1.0",
    description="LangGraph Studio Web integration for AI Politician",
    author="AI Politician Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langgraph-cli[inmem]>=0.0.19",
        "fastapi>=0.104.0",
        "uvicorn>=0.23.2",
    ],
)
