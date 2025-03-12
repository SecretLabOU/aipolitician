from setuptools import setup, find_packages

setup(
    name="aipolitician",
    version="0.1.0",
    description="AI models simulating political figures' speech patterns",
    author="AI Politician Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "torch>=2.0.0",
        "transformers>=4.37.0",
        "peft>=0.6.0",
    ],
    extras_require={
        "scraper": ["requests", "beautifulsoup4", "selenium"],
        "training": ["datasets", "accelerate", "bitsandbytes"],
        "chat": ["gradio", "sentence-transformers"],
    },
) 