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

3. Run the GPU setup script:
```bash
./scripts/run_on_gpu.sh <your-session-name>
```

The script will:
- Check GPU availability
- Set up a virtual environment
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

2. Or manually deactivate your session:
```bash
genv deactivate --id <your-session-name>
```

### Best Practices for GPU Usage

1. Resource Management:
   - Always clean up your GPU sessions when done
   - Monitor GPU memory usage with `nvidia-smi`
   - Use only the GPUs you need

2. Performance Optimization:
   - Set appropriate batch sizes based on GPU memory
   - Enable mixed precision training when possible
   - Monitor GPU utilization to ensure efficient use

3. Troubleshooting:
   - If GPU memory issues occur, try reducing batch size
   - Check for orphaned sessions with `genv devices`
   - Use `nvidia-smi` to monitor GPU health

[Rest of the README content remains the same]
