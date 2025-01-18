# Project Setup and Execution

## Prerequisites

### Docker

Ensure Docker is installed and running on your system. Follow the official installation guide for your operating system:

- [Install Docker](https://www.docker.com/products/docker-desktop)

To verify the installation, run:

```bash
docker --version
```

---

## Steps to Get Started

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/natalie-a-1/PoliticianAI.git
cd PoliticianAI
```

### 2. Create a `.env` File

Create a `.env` file in the root directory of the project to store your API key securely. Add the following content:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

> **Important:** The `.env` file is excluded from version control via `.gitignore` to protect your API key.

### 3. Build the Docker Image

Build the Docker image using the `Dockerfile` provided in the repository:

```bash
docker build -t politician-ai .
```

This command will create a Docker image named `politician-ai`.

### 4. Run the Docker Container

Run the application in a container while securely passing the `.env` file:

```bash
docker run -it --rm --env-file .env politician-ai
```

---

## Directory Structure

Ensure your project structure looks like this:

```
.
├── agent_workspace/       # Optional workspace directory (if needed)
├── .env                   # Contains environment variables (e.g., API key)
├── .gitignore             # Excludes .env from version control
├── Dockerfile             # Defines the Docker image
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── main.py                # Main script to run
```

---

## Troubleshooting

### Common Issues
- **Docker not found**: Ensure Docker is installed and running.
- **API key not set**: Verify that the `.env` file exists and contains the correct `OPENAI_API_KEY`.

For additional support, consult the [Docker documentation](https://docs.docker.com/).