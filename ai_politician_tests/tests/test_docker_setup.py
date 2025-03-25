import pytest
import docker
import subprocess
import re
from typing import Tuple, Optional

def test_docker_running():
    """Test that Docker is running and accessible via the Docker SDK."""
    try:
        # Create a Docker client
        client = docker.from_env()
        
        # Attempt to ping the Docker daemon
        ping_result = client.ping()
        
        # Check if the ping was successful
        assert ping_result is True, "Docker daemon is not responding"
        
        # Get Docker info to further verify connectivity
        docker_info = client.info()
        assert docker_info["Name"], "Could not retrieve Docker host information"
        
        print(f"Docker is running. Docker version: {docker_info.get('ServerVersion', 'Unknown')}")
        
    except docker.errors.DockerException as e:
        pytest.fail(f"Docker daemon is not accessible: {str(e)}")

def _parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse version string into a tuple of integers (major, minor, patch)."""
    # Extract version numbers using regex
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)  # Default if parsing fails

def _run_command(command: str) -> Optional[str]:
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Command '{command}' failed with error: {e}")
        return None

def test_docker_compose_installed():
    """Test that Docker Compose is installed and reports the correct version."""
    # Run docker compose version command
    version_output = _run_command("docker compose version")
    
    # Verify that the command output contains version information
    assert version_output, "Docker Compose version command returned no output"
    print(f"Docker Compose version output: {version_output}")
    
    # Extract the version number
    version_match = re.search(r'v?(\d+\.\d+\.\d+)', version_output)
    assert version_match, "Could not find Docker Compose version number in output"
    
    version_str = version_match.group(1)
    version_tuple = _parse_version(version_str)
    
    # Check that the version is at least 2.0.0
    min_version = (2, 0, 0)
    assert version_tuple >= min_version, f"Docker Compose version {version_str} is less than minimum required version 2.0.0"
    
    print(f"Docker Compose version {version_str} meets the minimum requirement of 2.0.0")

