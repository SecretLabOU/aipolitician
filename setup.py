from setuptools import setup, find_packages

setup(
    name="politician_ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Core Framework
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        
        # LangChain and Vector Store
        "langchain==0.0.352",
        "faiss-cpu==1.7.4",
        "chromadb==0.4.22",
        
        # Machine Learning
        "torch==2.1.1",
        "transformers==4.35.2",
        "sentencepiece==0.1.99",
        "accelerate==0.25.0",
        
        # Database
        "sqlalchemy==2.0.23",
        "aiosqlite==0.19.0",
        
        # Data Processing
        "pandas==1.5.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        
        # Utilities
        "python-dotenv==1.0.0",
        "pydantic==2.5.2",
        "tqdm==4.66.1",
        "requests==2.31.0",
        "python-multipart==0.0.6",
    ],
    python_requires=">=3.8",
    description="AI system for political discourse simulation",
    author="PoliticianAI Contributors",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/PoliticianAI",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
