# CLAUDE.md - AI Politician LangGraph Project

## Build/Test/Lint Commands
```bash
# Run all tests
make test

# Run a single test file
make test TEST_FILE=tests/unit_tests/test_configuration.py

# Run tests with profiling
make test_profile

# Lint code
make lint

# Lint only modified files
make lint_diff

# Format code
make format

# Spell check
make spell_check
```

## Code Style Guidelines
- **Typing**: Use strict typing with mypy. All functions require type annotations.
- **Docstrings**: Follow Google docstring style convention as enforced by ruff.
- **Imports**: Sort imports with isort via ruff (`make format`).
- **Error Handling**: Use custom error classes and the error_handler decorator.
- **Naming**: snake_case for variables/functions, PascalCase for classes.
- **Structure**: Use dataclasses for state management, especially with LangGraph.
- **Async**: Prefer async functions for LLM calls and I/O operations.
- **Dependencies**: Manage with pyproject.toml, minimum versions in requirements.txt.
- **LangGraph**: Follow graph-based architecture patterns with typed state schemas.