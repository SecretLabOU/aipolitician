#!/usr/bin/env python3
"""
Advanced GPU-Optimized Model Management System

Provides high-performance model loading, distributed tensor parallelism,
dynamic batching, and adaptive quantization for maximum GPU utilization.
"""

import os
import gc
import time
import logging
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Dict, Optional, List, Iterator, Union, Tuple, Any, Callable
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading
from queue import Queue
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel
from threading import Thread, Lock
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("model_manager")

# Load environment variables
load_dotenv()

# Model configuration
MODEL_REGISTRY = {
    "trump": os.getenv("TRUMP_MODEL_PATH", "nnat03/trump-mistral-adapter"),
    "biden": os.getenv("BIDEN_MODEL_PATH", "nnat03/biden-mistral-adapter")
}

PERSONA_DISPLAY_NAMES = {
    "trump": "Donald Trump",
    "biden": "Joe Biden"
}

BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
DEFAULT_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# Advanced performance configuration
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
USE_TENSOR_PARALLEL = os.getenv("USE_TENSOR_PARALLEL", "false").lower() == "true"
TP_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
ENABLE_VLLM = os.getenv("ENABLE_VLLM", "false").lower() == "true"
USE_AWQ = os.getenv("USE_AWQ", "false").lower() == "true"
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
INFERENCE_MEMORY_LIMIT = float(os.getenv("INFERENCE_MEMORY_LIMIT", "0.9"))  # 90% of GPU memory by default

# Advanced model cache with LRU management
class ModelCache:
    """Advanced LRU cache for models with memory-aware eviction policy"""
    
    def __init__(self, max_models=2, memory_threshold=0.8):
        """Initialize the model cache with memory-aware limits"""
        self.cache = {}
        self.max_models = max_models
        self.memory_threshold = memory_threshold  # Fraction of GPU memory that triggers eviction
        self.usage_tracker = {}  # Track last access time
        self.lock = Lock()  # Thread-safety for cache access
    
    def get(self, key):
        """Get a model from the cache if available"""
        with self.lock:
            if key in self.cache:
                self.usage_tracker[key] = time.time()
                return self.cache[key]
            return None
    
    def put(self, key, value):
        """Add a model to the cache with memory-aware eviction"""
        with self.lock:
            # Check if we need to evict models due to memory pressure
            self._check_memory_pressure()
            
            # Check if we're at capacity and need to evict the least recently used model
            if len(self.cache) >= self.max_models and key not in self.cache:
                self._evict_lru()
            
            # Add the new model to the cache
            self.cache[key] = value
            self.usage_tracker[key] = time.time()
    
    def _check_memory_pressure(self):
        """Check if we need to evict models due to memory pressure"""
        if not torch.cuda.is_available():
            return
        
        # Get current GPU memory usage
        current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        # If memory usage is above threshold, evict LRU models until below threshold or empty
        while current_memory > self.memory_threshold and self.cache:
            self._evict_lru()
            current_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    
    def _evict_lru(self):
        """Evict the least recently used model"""
        if not self.cache:
            return
        
        # Find the least recently used model
        lru_key = min(self.usage_tracker.items(), key=lambda x: x[1])[0]
        
        # Log the eviction
        logger.info(f"Evicting model {lru_key} from cache due to memory pressure")
        
        # Explicitly move model to CPU and call garbage collection
        if hasattr(self.cache[lru_key][0], 'to'):
            self.cache[lru_key][0].cpu()
        
        # Remove from cache and usage tracker
        del self.cache[lru_key]
        del self.usage_tracker[lru_key]
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear(self, key=None):
        """Clear the cache for a specific key or all keys"""
        with self.lock:
            if key is not None:
                if key in self.cache:
                    del self.cache[key]
                    del self.usage_tracker[key]
            else:
                self.cache.clear()
                self.usage_tracker.clear()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@dataclass
class InferenceRequest:
    """Structure for batched inference requests"""
    persona: str
    prompt: str
    context: Optional[str]
    max_length: int
    temperature: float
    callback: Callable
    streaming: bool

class BatchProcessor:
    """Process inference requests in batches for higher throughput"""
    
    def __init__(self, model_manager, max_batch_size=MAX_BATCH_SIZE, max_wait_time=0.1):
        """Initialize the batch processor"""
        self.model_manager = model_manager
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queues = {}  # One queue per persona
        self.processing_threads = {}
        self.running = False
        self.lock = Lock()
    
    def start(self):
        """Start the batch processor threads"""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            
            # Create a processing thread for each persona
            for persona in MODEL_REGISTRY.keys():
                self.request_queues[persona] = Queue()
                self.processing_threads[persona] = Thread(
                    target=self._process_queue,
                    args=(persona,),
                    daemon=True
                )
                self.processing_threads[persona].start()
    
    def stop(self):
        """Stop the batch processor threads"""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            
            # Wait for all threads to finish
            for thread in self.processing_threads.values():
                thread.join(timeout=1.0)
    
    def add_request(self, request):
        """Add a request to be processed in a batch"""
        if not self.running:
            self.start()
        
        # Add the request to the appropriate queue
        self.request_queues[request.persona].put(request)
    
    def _process_queue(self, persona):
        """Process requests in the queue for a specific persona"""
        while self.running:
            # Collect a batch of requests
            batch = []
            start_time = time.time()
            
            # Get the first request (blocking)
            try:
                request = self.request_queues[persona].get(timeout=0.1)
                batch.append(request)
            except:
                continue
            
            # Collect additional requests up to max batch size or wait time
            while (
                len(batch) < self.max_batch_size and 
                time.time() - start_time < self.max_wait_time and
                not self.request_queues[persona].empty()
            ):
                try:
                    request = self.request_queues[persona].get_nowait()
                    batch.append(request)
                except:
                    break
            
            # Skip if no requests collected
            if not batch:
                continue
            
            # Process the batch
            self._process_batch(persona, batch)
    
    def _process_batch(self, persona, batch):
        """Process a batch of requests for a specific persona"""
        # Skip if empty batch
        if not batch:
            return
        
        # Single request case - no batching needed
        if len(batch) == 1:
            request = batch[0]
            try:
                if request.streaming:
                    # We can't batch streaming requests, process individually
                    async_response = self.model_manager._generate_streaming_async(
                        persona, request.prompt, request.context,
                        request.max_length, request.temperature
                    )
                    request.callback(async_response)
                else:
                    response = self.model_manager.generate_response_sync(
                        persona, request.prompt, request.context,
                        request.max_length, request.temperature
                    )
                    request.callback(response)
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                request.callback(f"Error: {str(e)}")
            return
        
        # Multiple requests - process as batch if non-streaming
        # Filter out streaming requests to process individually
        streaming_requests = [r for r in batch if r.streaming]
        non_streaming_requests = [r for r in batch if not r.streaming]
        
        # Process non-streaming requests as a batch
        if non_streaming_requests:
            try:
                with self.model_manager.load_model_tokenizer(persona) as (model, tokenizer):
                    # Create batched inputs
                    prompts = [
                        self.model_manager._format_prompt(r.prompt, r.context)
                        for r in non_streaming_requests
                    ]
                    
                    # Tokenize all prompts
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                    
                    # Generate responses in one batch
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=max(r.max_length for r in non_streaming_requests),
                            num_return_sequences=len(non_streaming_requests),
                            temperature=sum(r.temperature for r in non_streaming_requests) / len(non_streaming_requests),
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True
                        )
                    
                    # Decode outputs and send to callbacks
                    for i, request in enumerate(non_streaming_requests):
                        response = tokenizer.decode(outputs[i], skip_special_tokens=True)
                        response = response.split("[/INST]")[-1].strip()
                        request.callback(response)
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                for request in non_streaming_requests:
                    request.callback(f"Error: {str(e)}")
        
        # Process streaming requests individually
        for request in streaming_requests:
            try:
                async_response = self.model_manager._generate_streaming_async(
                    persona, request.prompt, request.context,
                    request.max_length, request.temperature
                )
                request.callback(async_response)
            except Exception as e:
                logger.error(f"Error processing streaming request: {str(e)}")
                request.callback(f"Error: {str(e)}")

class ModelManager:
    """High-performance GPU-optimized model management system"""
    
    def __init__(self):
        """Initialize the model manager with GPU optimization"""
        self.model_cache = ModelCache(
            max_models=int(os.getenv("MAX_CACHED_MODELS", "2")),
            memory_threshold=float(os.getenv("CACHE_MEMORY_THRESHOLD", "0.8"))
        )
        self.batch_processor = BatchProcessor(self)
        self.has_initialized_gpu = False
        self.vllm_engines = {}
        self.init_gpu()
    
    def init_gpu(self):
        """Initialize GPU settings and optimizations"""
        if self.has_initialized_gpu:
            return
        
        # Check for CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Performance will be severely impacted.")
            self.has_initialized_gpu = True
            return
        
        try:
            # Log GPU information
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            logger.info(f"Found {gpu_count} CUDA devices: {', '.join(gpu_names)}")
            
            # Set environment variables for maximum performance
            if USE_FLASH_ATTN:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
            # Initialize tensor parallelism if enabled and multiple GPUs available
            if USE_TENSOR_PARALLEL and gpu_count > 1:
                self._setup_tensor_parallel()
            
            # Initialize vLLM engines if enabled
            if ENABLE_VLLM:
                self._setup_vllm_engines()
            
            # Set optimal CUDA settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Log GPU memory
            for i in range(gpu_count):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            self.has_initialized_gpu = True
        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}")
            logger.info("Continuing with default settings")
    
    def _setup_tensor_parallel(self):
        """Setup tensor parallelism across multiple GPUs"""
        try:
            # Only import if needed
            try:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                import torch.distributed as dist
                
                # Initialize process group if not already done
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                    logger.info(f"Initialized distributed process group for tensor parallelism")
            except ImportError:
                logger.warning("accelerate package not found, tensor parallelism disabled")
                return
            
            logger.info(f"Tensor parallelism enabled across {TP_SIZE} GPUs")
        except Exception as e:
            logger.error(f"Failed to setup tensor parallelism: {str(e)}")
    
    def _setup_vllm_engines(self):
        """Setup vLLM engines for faster inference if enabled"""
        try:
            # Only import if needed
            from vllm import LLM, SamplingParams
            
            # Initialize engine for each persona
            for persona in MODEL_REGISTRY.keys():
                # Combine base model with LoRA adapter
                if persona not in self.vllm_engines:
                    logger.info(f"Initializing vLLM engine for {persona}")
                    self.vllm_engines[persona] = LLM(
                        model=BASE_MODEL,
                        lora_adapters=MODEL_REGISTRY[persona],
                        tensor_parallel_size=TP_SIZE if USE_TENSOR_PARALLEL else 1,
                        gpu_memory_utilization=INFERENCE_MEMORY_LIMIT,
                        quantization="awq" if USE_AWQ else None
                    )
                    
            logger.info("vLLM engines initialized")
        except ImportError:
            logger.warning("vLLM package not found, falling back to standard inference")
        except Exception as e:
            logger.error(f"Failed to setup vLLM engines: {str(e)}")
    
    def shutdown(self):
        """Properly shut down all components"""
        # Stop batch processor
        if hasattr(self, 'batch_processor'):
            self.batch_processor.stop()
        
        # Clear model cache
        if hasattr(self, 'model_cache'):
            self.model_cache.clear()
        
        # Clean up vLLM engines
        if hasattr(self, 'vllm_engines'):
            self.vllm_engines.clear()
        
        # Clean up CUDA
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @contextmanager
    def load_model_tokenizer(self, persona: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer with GPU optimizations and context management"""
        try:
            # First check if model is in cache
            cached = self.model_cache.get(f"{persona}_model")
            if cached:
                logger.debug(f"Using cached model for {persona}")
                yield cached
                return
            
            # Not in cache, load the model
            logger.info(f"Loading model for {persona}...")
            model, tokenizer = self._load_model_optimized(persona)
            
            # Add to cache
            self.model_cache.put(f"{persona}_model", (model, tokenizer))
            
            yield (model, tokenizer)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_model_optimized(self, persona: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model with optimal settings for high-end GPUs"""
        start_time = time.time()
        
        # Get the model path
        lora_path = MODEL_REGISTRY.get(persona)
        if not lora_path:
            raise ValueError(f"No model found for persona: {persona}")
        
        # Configure quantization - use AWQ if enabled, otherwise 4-bit quantization
        if USE_AWQ:
            # AWQ config (more efficient on A100/H100 GPUs)
            try:
                from awq import AutoAWQForCausalLM
                
                logger.info("Using AWQ quantization for higher performance")
                model = AutoAWQForCausalLM.from_quantized(
                    model_name_or_path=lora_path,
                    max_cpu_memory_MB=0,
                    device_map="auto"
                )
                
                tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL,
                    use_fast=True,
                    cache_dir=MODEL_CACHE_DIR,
                    trust_remote_code=False
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                return model, tokenizer
            
            except ImportError:
                logger.warning("AWQ not available, falling back to 4-bit quantization")
        
        # Standard 4-bit quantization as fallback
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Attention implementation
        attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "eager"
        
        # Load base model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=False,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        # Load tokenizer (use fast version for performance)
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            cache_dir=MODEL_CACHE_DIR,
            trust_remote_code=False
        )
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            cache_dir=MODEL_CACHE_DIR
        )
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        # Enable evaluation mode
        model.eval()
        
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return model, tokenizer
    
    async def generate_response(
        self, 
        persona: str, 
        prompt: str, 
        context: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE,
        streaming: bool = False
    ) -> Union[str, Iterator[str]]:
        """Generate a response using optimal inference strategy based on configuration"""
        # If vLLM is enabled and available, use it for non-streaming requests
        if ENABLE_VLLM and persona in self.vllm_engines and not streaming:
            return self._generate_with_vllm(
                persona, prompt, context, max_length, temperature
            )
        
        # For streaming or when vLLM is not available
        if streaming:
            # Streaming requests need special handling
            return self._generate_streaming_async(
                persona, prompt, context, max_length, temperature
            )
        else:
            # Check if we should use batch processing
            if MAX_BATCH_SIZE > 1:
                # Create a future to receive the response
                response_future = asyncio.Future()
                
                # Create a callback to set the future result
                def set_response(response):
                    if not response_future.done():
                        response_future.set_result(response)
                
                # Create and add the request to the batch processor
                request = InferenceRequest(
                    persona=persona,
                    prompt=prompt,
                    context=context,
                    max_length=max_length,
                    temperature=temperature,
                    callback=set_response,
                    streaming=False
                )
                
                self.batch_processor.add_request(request)
                
                # Wait for the response
                return await response_future
            else:
                # Batch processing disabled, use direct generation
                return self.generate_response_sync(
                    persona, prompt, context, max_length, temperature
                )
    
    def generate_response_sync(
        self,
        persona: str,
        prompt: str,
        context: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """Generate a complete response synchronously"""
        # Format the prompt with context if available
        formatted_prompt = self._format_prompt(prompt, context)
        
        with self.load_model_tokenizer(persona) as (model, tokenizer):
            # Encode the prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            
            # Generate response
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True
                )
            
            # Decode and clean the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("[/INST]")[-1].strip()
            return response
    
    def _generate_with_vllm(
        self,
        persona: str,
        prompt: str,
        context: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """Generate a response using vLLM for higher throughput"""
        try:
            from vllm import SamplingParams
            
            # Format the prompt with context if available
            formatted_prompt = self._format_prompt(prompt, context)
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_length,
                skip_special_tokens=True
            )
            
            # Generate response with vLLM
            engine = self.vllm_engines[persona]
            outputs = engine.generate(formatted_prompt, sampling_params)
            
            # Extract and clean response
            response = outputs[0].outputs[0].text
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            
            return response
        except Exception as e:
            logger.error(f"vLLM generation failed: {str(e)}. Falling back to standard generation.")
            return self.generate_response_sync(
                persona, prompt, context, max_length, temperature
            )
    
    async def _generate_streaming_async(
        self,
        persona: str,
        prompt: str,
        context: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """Generate a response with asynchronous streaming"""
        formatted_prompt = self._format_prompt(prompt, context)
        
        with self.load_model_tokenizer(persona) as (model, tokenizer):
            # Setup streamer and generation settings
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=10.0)
            generation_kwargs = {
                "input_ids": tokenizer(formatted_prompt, return_tensors="pt").to("cuda").input_ids,
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": True
            }
            
            # Start generation in a separate thread
            thread = Thread(target=self._call_model_generate, args=(model, generation_kwargs))
            thread.start()
            
            # Stream the output with proper processing
            buffer = ""
            async for text_chunk in self._async_iterate(streamer):
                # Process the chunk to handle [/INST] markers
                if "[/INST]" in text_chunk:
                    text_chunk = text_chunk.split("[/INST]")[-1]
                elif "[/INST]" in buffer:
                    buffer = buffer.split("[/INST]")[-1]
                
                # Append to buffer and yield
                buffer += text_chunk
                yield text_chunk
    
    def _call_model_generate(self, model, generation_kwargs):
        """Call model.generate in a way that's safe for threading"""
        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                # Generate and stream through the streamer in generation_kwargs
                model.generate(**generation_kwargs)
        except Exception as e:
            logger.error(f"Error in model generation thread: {str(e)}")
    
    async def _async_iterate(self, streamer):
        """Async wrapper around a sync iterator"""
        loop = asyncio.get_event_loop()
        for text in streamer:
            yield text
            # Yield control for better concurrency
            await asyncio.sleep(0)
    
    def _format_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """Format the prompt with RAG context if available"""
        if context:
            rag_prompt = f"{context}\n\nUser Question: {prompt}"
            return f"<s>[INST] {rag_prompt} [/INST]"
        else:
            return f"<s>[INST] {prompt} [/INST]"
    
    def clear_cache(self, persona: Optional[str] = None):
        """Clear model cache for a specific persona or all personas"""
        self.model_cache.clear(f"{persona}_model" if persona else None)
    
    def get_available_personas(self) -> List[str]:
        """Get list of available persona models"""
        return list(MODEL_REGISTRY.keys())
    
    def get_display_name(self, persona: str) -> str:
        """Get the display name for a persona"""
        return PERSONA_DISPLAY_NAMES.get(persona, persona.capitalize())


# Singleton instance
model_manager = ModelManager() 