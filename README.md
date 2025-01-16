# Project Setup and Execution

This README provides the steps to set up and run the `test.py` script.&#x20;

---

## Prerequisites

### Python Version

Ensure that you have **Python 3.10** installed on your system.

> **Note:** Using Python 3.12 may cause compatibility issues.

### Install `pip`

`pip` is required to manage Python packages. Most Python installations include it by default, but verify by running:

```bash
pip --version
```

If not installed, follow the official instructions [here](https://pip.pypa.io/en/stable/installation/).

---

## Steps to Get Started

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Set Up Virtual Environment

To isolate the dependencies, create and activate a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate   # For Windows
```

### 3. Install Dependencies

Install the required dependencies from the existing `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the root directory of the project and add the following variables:

```dotenv
WORKSPACE_DIR=agent_workspace
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### 5. Verify Directory Structure

Ensure the following directory structure exists:

```
.
├── agent_workspace
├── venv
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── test.py
```

### 6. Run the Script

To run `test.py`, execute the following command:

```bash
python test.py
```
