"""Setup script for PoliticianAI."""

import os
from setuptools import find_packages, setup

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    try:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        # Return empty list if file doesn't exist
        pass
    return requirements

# Read long description
def read_long_description():
    """Read README.md as long description."""
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Project metadata
NAME = 'politician_ai'
DESCRIPTION = 'AI system for political discourse simulation'
AUTHOR = 'PoliticianAI Contributors'
AUTHOR_EMAIL = 'contributors@politicianai.org'
URL = 'https://github.com/yourusername/PoliticianAI'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.8'

# Version handling
try:
    from setuptools_scm import get_version
    version = get_version()
except Exception:
    version = '1.0.0'

# Setup configuration
setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    
    # Package structure
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'dev': [
            'black==23.11.0',
            'isort==5.12.0',
            'mypy==1.7.1',
            'flake8==6.1.0',
            'pylint==3.0.2',
            'pre-commit==3.5.0'
        ],
        'test': [
            'pytest==7.4.3',
            'pytest-cov==4.1.0',
            'pytest-asyncio==0.21.1',
            'pytest-mock==3.12.0',
            'pytest-env==1.0.1'
        ],
        'docs': [
            'sphinx==7.2.6',
            'sphinx-rtd-theme==1.3.0',
            'sphinx-autodoc-typehints==1.24.0'
        ]
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'politician-ai=src.main:main',
        ],
    },
    
    # Package metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[
        'artificial intelligence',
        'natural language processing',
        'political discourse',
        'machine learning',
    ],
    
    # Project URLs
    project_urls={
        'Bug Reports': f'{URL}/issues',
        'Source': URL,
    },
)
