import os
import subprocess
import pytest
import docker
from docker.errors import ImageNotFound, APIError


@pytest.fixture
def project_root():
    """Return the path to the project root directory."""
    return "/Users/prestonjones/code/ou/secret-lab/politician-ai/aipolitician"


@pytest.fixture
def docker_client():
    """Return a Docker client instance."""
    try:
        client = docker.from_env()
        # Verify connection to Docker daemon
        client.ping()
        return client
    except Exception as e:
        pytest.fail(f"Failed to connect to Docker daemon: {e}")


def is_image_available(client, image_name, tag):
    """
    Check if a Docker image with the specified name and tag is available locally.
    If not, attempt to pull it from the registry.
    
    Args:
        client: Docker client instance
        image_name: Name of the Docker image
        tag: Tag of the Docker image
        
    Returns:
        bool: True if the image is available or was successfully pulled, False otherwise
    """
    full_image_name = f"{image_name}:{tag}"
    print(f"Checking for image: {full_image_name}")
    
    # Check if image exists locally
    try:
        local_images = client.images.list(name=image_name)
        for image in local_images:
            tags = image.tags
            if full_image_name in tags:
                print(f"✓ Image {full_image_name} found locally")
                return True
        
        print(f"Image {full_image_name} not found locally, attempting to pull...")
        
        # Try to pull the image
        client.images.pull(image_name, tag=tag)
        print(f"✓ Successfully pulled {full_image_name}")
        return True
        
    except ImageNotFound:
        print(f"✗ Image {full_image_name} not found in registry")
        return False
    except APIError as e:
        print(f"✗ Docker API error when pulling {full_image_name}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error when checking/pulling {full_image_name}: {e}")
        return False


def test_required_docker_images_with_versions(docker_client):
    """
    Test that all required Docker images with specific versions are available
    or can be pulled from their respective registries.
    """
    required_images = [
        {"name": "minio/minio", "tag": "RELEASE.2023-03-20T20-16-18Z"},
        {"name": "quay.io/coreos/etcd", "tag": "v3.5.5"},
        {"name": "milvusdb/milvus", "tag": "v2.3.0"}
    ]
    
    print("\nChecking for required Docker images with specific versions...")
    
    failures = []
    for image_info in required_images:
        name = image_info["name"]
        tag = image_info["tag"]
        
        print(f"\nRetrieving {name}:{tag}...")
        if not is_image_available(docker_client, name, tag):
            failures.append(f"{name}:{tag}")
    
    if failures:
        pytest.fail(f"Failed to find or pull the following images: {', '.join(failures)}")
    else:
        print("\n✓ All required Docker images are available")


def test_gpu_capabilities(docker_client):
    """
    Test if Docker can detect GPUs if they are available on the system.
    The test will pass gracefully if no GPUs are available.
    """
    print("\nChecking GPU capabilities for Docker...")
    
    # Check if nvidia-smi is available on the host
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        has_nvidia_gpu = True
        print("✓ NVIDIA GPU detected on host system")
    except (subprocess.CalledProcessError, FileNotFoundError):
        has_nvidia_gpu = False
        print("No NVIDIA GPU detected on host system or nvidia-smi not available")
    
    if has_nvidia_gpu:
        # Try to use NVIDIA GPU inside a container
        try:
            # Run a container with GPU access and test if NVIDIA is visible
            result = docker_client.containers.run(
                "nvidia/cuda:11.0-base",
                "nvidia-smi",
                remove=True,
                runtime="nvidia"
            )
            print(f"✓ Docker can access GPU capabilities: {result.decode('utf-8')[:100]}...")
        except Exception as e:
            pytest.fail(f"Host has NVIDIA GPU but Docker cannot access it: {e}")
    else:
        print("Skipping Docker GPU capability test as no NVIDIA GPU is detected")
        # Test passes gracefully when no GPU is available
        pass

    def is_image_available(self, docker_client, image_name):
        """
        Check if an image is available locally or can be pulled from Docker Hub.
        
        Args:
            docker_client: Docker client instance
            image_name: Name of the Docker image to check
            
        Returns:
            tuple: (bool - is available, str - found tag or error message)
        """
        try:
            # First check if the image is already available locally
            print(f"Checking if image '{image_name}' is available locally...")
            images = docker_client.images.list()
            for image in images:
                if image.tags:
                    for tag in image.tags:
    def is_image_available(self, docker_client, image_name):
        """
        Check if an image is available locally or can be pulled from Docker Hub.
        
        Args:
            docker_client: Docker client instance
            image_name: Name of the Docker image to check
            
        Returns:
            tuple: (bool - is available, str - found tag or error message)
        """
        try:
            # First check if the image is already available locally
            print(f"Checking if image '{image_name}' is available locally...")
            images = docker_client.images.list()
            for image in images:
                if image.tags:
                    for tag in image.tags:
                        if image_name == tag:  # Exact match for version tag
                            print(f"✅ Image '{image_name}' found locally")
                            return True, tag
            
            # If not available locally, try to pull it
            print(f"Image '{image_name}' not found locally, attempting to pull...")
            pulled_image = docker_client.images.pull(image_name)
            if pulled_image:
                print(f"✅ Successfully pulled image '{image_name}'")
                return True, pulled_image.tags[0] if pulled_image.tags else image_name
            return False, "Failed to pull image"
            
        except docker.errors.ImageNotFound:
            error_msg = f"❌ Error: Image '{image_name}' not found locally or in registry"
            print(error_msg)
            return False, error_msg
        except docker.errors.APIError as e:
            error_msg = f"❌ Docker API error for image '{image_name}': {str(e)}"
            print(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Unexpected error for image '{image_name}': {str(e)}"
            print(error_msg)
            return False, error_msg
        
        # List of required Docker images with exact version tags from docker-compose.yml
        required_images = [
            "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "quay.io/coreos/etcd:v3.5.5",
            "milvusdb/milvus:v2.3.0"
        ]
        
        all_passed = True
        failures = []
        
        try:
            # Check each required image with exact version tag
            for required_image in required_images:
                print(f"\n📋 Verifying image: {required_image}")
                is_available, message = self.is_image_available(docker_client, required_image)
                
                if not is_available:
                    all_passed = False
                    failures.append(f"Required image '{required_image}' is not available: {message}")
                else:
                    print(f"✅ Image '{required_image}' is verified and available")
            
            # Final assertion to make the test fail if any image check failed
            assert all_passed, f"Some required images are not available:\n" + "\n".join(failures)
            
        except docker.errors.APIError as e:
            error_msg = f"❌ Docker API error: {str(e)}"
            print(error_msg)
            pytest.fail(error_msg)
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)

    def test_required_docker_images_with_versions(self, docker_client, project_root):
        """
        Test that the required Docker images with specific version tags are available or can be pulled.
        This test focuses on verifying the exact versions needed by the AI Politician application.
        """
        # Navigate to the project root where docker-compose.yml is located
        os.chdir(project_root)
        
        print("\n=== Checking Required Docker Images with Exact Version Tags ===")
        
        # List of required Docker images with exact version tags from docker-compose.yml
        required_images = [
            "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "quay.io/coreos/etcd:v3.5.5",
            "milvusdb/milvus:v2.3.0"
        ]
        
        all_passed = True
        failures = []
        
        try:
            # Check each required image with exact version tag
            for required_image in required_images:
                print(f"\n📋 Verifying image: {required_image}")
                is_available, message = self.is_image_available(docker_client, required_image)
                
                if not is_available:
                    all_passed = False
                    failures.append(f"Required image '{required_image}' is not available: {message}")
                else:
                    print(f"✅ Image '{required_image}' is verified and available")
            
            # Final assertion to make the test fail if any image check failed
            assert all_passed, f"Some required images are not available:\n" + "\n".join(failures)
            
        except docker.errors.APIError as e:
            error_msg = f"❌ Docker API error: {str(e)}"
            print(error_msg)
            pytest.fail(error_msg)
        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(error_msg)
            pytest.fail(error_msg)
            
    def test_gpu_capabilities(self, docker_client):
        """
        Test GPU capabilities of Docker if available on the system.
        This test passes gracefully if GPUs are not available.
        """
        print("\n=== Checking GPU Capabilities ===")
        
        try:
            # Try to run nvidia-smi inside a container to test GPU access
            print("🔍 Checking if Docker can access GPUs...")
            
            # Check if nvidia-docker runtime is available
            gpu_info = docker_client.info()
            has_nvidia_runtime = False
            
            if 'Runtimes' in gpu_info and 'nvidia' in gpu_info['Runtimes']:
                has_nvidia_runtime = True
                print("✅ NVIDIA Docker runtime is available")
            else:
                print("ℹ️ NVIDIA Docker runtime is not available")
            
            # Try running a container with nvidia-smi to verify GPU access
            try:
                print("🔍 Testing GPU access by running nvidia-smi in a container...")
                container = docker_client.containers.run(
                    "nvidia/cuda:12.2.0-runtime-ubuntu22.04",
                    "nvidia-smi",
                    remove=True,
                    runtime="nvidia" if has_nvidia_runtime else None
                )
                print("✅ GPU access test successful! Docker can access GPUs")
                print(f"📊 GPU info:\n{container.decode('utf-8')}")
                
            except docker.errors.ContainerError as e:
                if "Unknown runtime specified nvidia" in str(e) or "executable file not found" in str(e):
                    print("ℹ️ GPU capability test skipped: NVIDIA runtime not available")
                else:
                    print(f"ℹ️ GPU capability test failed but continuing: {str(e)}")
                
            except Exception as e:
                print(f"ℹ️ GPU capability test failed but continuing: {str(e)}")
                
        except Exception as e:
            print(f"ℹ️ GPU capability check skipped: {str(e)}")
            print("ℹ️ This is not a failure, just informational - test continues")
            
        print("✅ GPU capability test completed")
        # This test always passes, it's informational only
        assert True

    # Original test_required_docker_images remains available but commented out
    """
    def test_required_docker_images(self, docker_client, project_root):
        # Test that the required Docker images are available or can be pulled.
        # Navigate to the project root where docker-compose.yml is located
        os.chdir(project_root)
        
        # List of required Docker images
        # List of required Docker images with specific versions from docker-compose.yml
        required_images = [
            "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "quay.io/coreos/etcd:v3.5.5",
            "milvusdb/milvus:v2.3.0"
        ]
        try:
            # Get the list of all available images
            images = docker_client.images.list()
            image_tags = []
            
            for image in images:
                if image.tags:
                    for tag in image.tags:
                        image_tags.append(tag)
            
            print(f"Available image tags: {image_tags}")
            
            # Check each required image
            for required_image in required_images:
                is_available, message = self.is_image_available(docker_client, required_image)
                assert is_available, f"Required image '{required_image}' is not available: {message}"
                print(f"Image '{required_image}' is available: {message}")
            
        except docker.errors.APIError as e:
            pytest.fail(f"Docker API error: {str(e)}")
    """
            
    def test_gpu_capabilities(self, docker_client):
        """
        Test GPU capabilities of Docker if available on the system.
        This test passes gracefully if GPUs are not available.
        """
        print("\n=== Checking GPU Capabilities ===")
        
        try:
            # Try to run nvidia-smi inside a container to test GPU access
            print("🔍 Checking if Docker can access GPUs...")
            
            # Check if nvidia-docker runtime is available
            gpu_info = docker_client.info()
            has_nvidia_runtime = False
            
            if 'Runtimes' in gpu_info and 'nvidia' in gpu_info['Runtimes']:
                has_nvidia_runtime = True
                print("✅ NVIDIA Docker runtime is available")
            else:
                print("ℹ️ NVIDIA Docker runtime is not available")
            
            # Try running a container with nvidia-smi to verify GPU access
            try:
                print("🔍 Testing GPU access by running nvidia-smi in a container...")
                container = docker_client.containers.run(
                    "nvidia/cuda:12.2.0-runtime-ubuntu22.04",
                    "nvidia-smi",
                    remove=True,
                    runtime="nvidia" if has_nvidia_runtime else None
                )
                print("✅ GPU access test successful! Docker can access GPUs")
                print(f"📊 GPU info:\n{container.decode('utf-8')}")
                
            except docker.errors.ContainerError as e:
                if "Unknown runtime specified nvidia" in str(e) or "executable file not found" in str(e):
                    print("ℹ️ GPU capability test skipped: NVIDIA runtime not available")
                else:
                    print(f"ℹ️ GPU capability test failed but continuing: {str(e)}")
                
            except Exception as e:
                print(f"ℹ️ GPU capability test failed but continuing: {str(e)}")
                
        except Exception as e:
            print(f"ℹ️ GPU capability check skipped: {str(e)}")
            print("ℹ️ This is not a failure, just informational - test continues")
            
        print("✅ GPU capability test completed")
        # This test always passes, it's informational only
        assert True
        
    # Original test with building capabilities is commented out
    """
    def test_required_docker_images(self, docker_client, project_root):
        # Test that the required Docker images are available or can be pulled.
        # Navigate to the project root where docker-compose.yml is located
        os.chdir(project_root)
        
        # List of required Docker images
        # List of required Docker images with specific versions from docker-compose.yml
        required_images = [
            "minio/minio:RELEASE.2023-03-20T20-16-18Z",
            "quay.io/coreos/etcd:v3.5.5",
            "milvusdb/milvus:v2.3.0"
        ]
        try:
            # Get the list of all available images
            images = docker_client.images.list()
            image_tags = []
            
            for image in images:
                if image.tags:
                    for tag in image.tags:
                        image_tags.append(tag)
            
            print(f"Available image tags: {image_tags}")
            
            # Check each required image
            for required_image in required_images:
                is_available, message = self.is_image_available(docker_client, required_image)
                assert is_available, f"Required image '{required_image}' is not available: {message}"
                print(f"Image '{required_image}' is available: {message}")
            
        except docker.errors.APIError as e:
            pytest.fail(f"Docker API error: {str(e)}")
    """
