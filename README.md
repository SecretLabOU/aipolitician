# PoliticianAI

[Previous content remains the same until the GPU Support section]

## GPU Support

The system automatically detects and uses available GPUs. For optimal performance:
- CUDA 11.4+
- 8GB+ VRAM
- Latest GPU drivers

### Running on GPU Server

1. SSH into the GPU server:
```bash
# For small jobs
ssh <username>

# For large jobs
ssh <username>
```

2. Clone and prepare the repository:
```bash
git clone https://github.com/yourusername/PoliticianAI.git
cd PoliticianAI
```

3. Initialize genv shell environment:
```bash
# Run the initialization script
./scripts/init_genv.sh

# Source your bashrc to apply changes
source ~/.bashrc
```

4. Run the GPU setup script with conda environment:
```bash
# If you want to use an existing conda environment:
./scripts/run_on_gpu.sh existing-env-name your-session-name

# Or to create a new conda environment:
./scripts/run_on_gpu.sh politician-ai your-session-name
```

The script will:
- Initialize genv shell environment
- Check GPU availability
- Set up/activate conda environment
- Install dependencies
- Download required models
- Initialize data
- Start the application

### Managing GPU Sessions

You can check GPU status at any time:
```bash
# View available GPUs and utilization
nvidia-smi

# Show GPUs attached to current sessions
genv devices
```

For manual GPU session management:
```bash
# Activate new session
genv activate --id <your-name>

# Attach GPU
genv attach --count 1
```

### Cleaning Up GPU Sessions

When you're done with your work:

1. Use the cleanup script:
```bash
./scripts/cleanup_gpu.sh
```
This script will:
- Show current GPU status
- List active sessions
- Let you deactivate specific or all sessions
- Display final GPU status

2. Deactivate your conda environment:
```bash
conda deactivate
```

### Troubleshooting GPU Setup

If you encounter issues:

1. Shell Initialization:
   - Ensure genv shell is initialized: `./scripts/init_genv.sh`
   - Verify initialization in ~/.bashrc
   - Try opening a new terminal session

2. Conda Environment Issues:
   - List environments: `conda env list`
   - Try creating a new environment: `conda create -n new-env python=3.8`
   - Check if conda is in PATH: `which conda`

3. GPU Session Issues:
   - Check GPU availability: `nvidia-smi`
   - List active sessions: `genv devices`
   - Try cleaning up old sessions: `./scripts/cleanup_gpu.sh`

4. Common Problems:
   - "Shell not properly initialized": Run `./scripts/init_genv.sh`
   - "Not running in active environment": Ensure conda environment is activated
   - "No GPUs available": Check if all GPUs are in use
   - "Conda command not found": Source your conda initialization

### Best Practices for GPU Usage

1. Environment Management:
   - Use conda environments for isolation
   - Keep environments clean and focused
   - Document environment dependencies

2. Resource Management:
   - Always clean up your GPU sessions when done
   - Monitor GPU memory usage with `nvidia-smi`
   - Use only the GPUs you need

3. Performance Optimization:
   - Set appropriate batch sizes based on GPU memory
   - Enable mixed precision training when possible
   - Monitor GPU utilization to ensure efficient use

[Rest of the README content remains the same]
