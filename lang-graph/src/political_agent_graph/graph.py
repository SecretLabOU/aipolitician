"""
High-performance political agent graph implementation.

Using advanced GPU parallelism and optimized LangGraph for maximum throughput.
"""

import json
import asyncio
import time
import logging
import os
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import local modules
from .state import ConversationState, get_initial_state
from .config import get_model_for_task, get_temperature_for_task
from .prompts import (
    analyze_sentiment_template,
    determine_topic_template,
    decide_deflection_template,
    generate_policy_stance_template,
    format_response_template,
    fact_check_template,
    retrieve_context_template,
)
from . import persona_manager

# Import RAG functionality if available
try:
    from db.utils.rag_utils import integrate_with_chat, enhance_query
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.info("RAG functionality not available. Operating in standalone mode.")

# Performance settings
ENABLE_PARALLEL = True  # Always use parallel processing for best performance
MAX_WORKERS = min(os.cpu_count() or 4, 8)  # Limit worker threads
MAX_RETRIES = 2         # Number of retries for failing nodes


# Thread pool for parallel CPU tasks
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)


async def analyze_sentiment(state: ConversationState) -> ConversationState:
    """Analyze sentiment of user input."""
    model = get_model_for_task("analyze_sentiment")
    temperature = get_temperature_for_task("analyze_sentiment")
    
    try:
        prompt = analyze_sentiment_template.format(user_input=state.user_input)
        sentiment = model.invoke(prompt, temperature=temperature).strip().lower()
        
        # Validate sentiment
        if sentiment not in ["positive", "negative", "neutral"]:
            sentiment = "neutral"
        
        state.topic_sentiment = sentiment
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        state.topic_sentiment = "neutral"
    
    return state


async def determine_topic(state: ConversationState) -> ConversationState:
    """Determine topic of user input."""
    model = get_model_for_task("determine_topic")
    temperature = get_temperature_for_task("determine_topic")
    
    try:
        prompt = determine_topic_template.format(user_input=state.user_input)
        topic = model.invoke(prompt, temperature=temperature).strip().lower()
        state.current_topic = topic
        
    except Exception as e:
        logger.error(f"Topic determination error: {e}")
        state.current_topic = "general politics"
    
    return state


async def retrieve_context(state: ConversationState) -> ConversationState:
    """Retrieve relevant context using RAG."""
    if not RAG_AVAILABLE:
        return state
    
    try:
        persona = persona_manager.get_active_persona()
        enhanced_query = enhance_query(state.user_input, persona["name"])
        
        context, success = integrate_with_chat(
            query=enhanced_query,
            persona=persona["name"],
            top_k=12,  # Increased for better coverage
            enable_reranking=True
        )
        
        state.retrieved_context = context if success and context else ""
        
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        state.retrieved_context = ""
    
    return state


async def process_context(state: ConversationState) -> ConversationState:
    """Process and summarize retrieved context."""
    if not state.retrieved_context:
        return state
    
    try:
        model = get_model_for_task("process_context")
        temperature = get_temperature_for_task("process_context")
        
        prompt = retrieve_context_template.format(
            context=state.retrieved_context[:2000],  # Increased context limit
            question=state.user_input
        )
        
        # Process context in background thread to avoid blocking
        loop = asyncio.get_event_loop()
        state.processed_context = await loop.run_in_executor(
            thread_pool,
            lambda: model.invoke(prompt, temperature=temperature).strip()
        )
        
    except Exception as e:
        logger.error(f"Context processing error: {e}")
        state.processed_context = state.retrieved_context[:500]  # Use truncated raw context as fallback
    
    return state


async def decide_deflection(state: ConversationState) -> ConversationState:
    """Decide whether to deflect the question."""
    model = get_model_for_task("decide_deflection")
    temperature = get_temperature_for_task("decide_deflection")
    
    try:
        persona = persona_manager.get_active_persona()
        
        # Use processed context if available
        context_summary = state.processed_context if hasattr(state, 'processed_context') else ""
        
        prompt = decide_deflection_template.format(
            politician_name=persona["name"],
            politician_party=persona["party"],
            user_input=state.user_input,
            current_topic=state.current_topic,
            topic_sentiment=state.topic_sentiment,
            rhetoric_style=json.dumps(persona["rhetorical_style"], indent=2),
            context=context_summary
        )
        
        result = model.invoke(prompt, temperature=temperature).strip()
        
        # Parse result
        lines = result.split("\n")
        state.should_deflect = lines[0].lower() == "true"
        
        # Get deflection topic if needed
        if state.should_deflect and len(lines) > 1:
            state.deflection_topic = lines[1].strip()
            
    except Exception as e:
        logger.error(f"Deflection decision error: {e}")
        state.should_deflect = False
        state.deflection_topic = None
    
    return state


async def generate_policy_stance(state: ConversationState) -> ConversationState:
    """Generate politician's policy stance."""
    model = get_model_for_task("generate_policy_stance")
    temperature = get_temperature_for_task("generate_policy_stance")
    
    try:
        persona = persona_manager.get_active_persona()
        
        # Determine topic
        topic = state.deflection_topic if state.should_deflect else state.current_topic
        
        # Get policy data
        policy_data = persona.get("policy_stances", {})
        relevant_policy = policy_data.get(topic, {})
        
        # Include context if available
        context_info = ""
        if hasattr(state, 'processed_context') and state.processed_context and not state.should_deflect:
            context_info = f"\nRelevant information:\n{state.processed_context}"
        elif state.retrieved_context and not state.should_deflect:
            context_info = f"\nRelevant information:\n{state.retrieved_context[:1000]}"
        
        prompt = generate_policy_stance_template.format(
            politician_name=persona["name"],
            politician_party=persona["party"],
            current_topic=topic,
            user_input=state.user_input,
            policy_stances=json.dumps(relevant_policy, indent=2) if relevant_policy else "No specific stance on this topic.",
            speech_patterns=json.dumps(persona["speech_patterns"], indent=2),
            context=context_info
        )
        
        # Generate policy stance using GPU acceleration
        state.policy_stance = model.invoke(prompt, temperature=temperature).strip()
        
    except Exception as e:
        logger.error(f"Policy stance error: {e}")
        persona = persona_manager.get_active_persona()
        topic = state.current_topic or "this issue"
        state.policy_stance = f"As {persona['name']}, I have clear views on {topic}."
    
    return state


async def fact_check(state: ConversationState) -> ConversationState:
    """Fact-check policy stance against context."""
    # Skip if no context or deflecting
    if not state.retrieved_context or state.should_deflect:
        state.fact_check_result = "SKIPPED"
        return state
    
    model = get_model_for_task("fact_check")
    temperature = get_temperature_for_task("fact_check")
    
    try:
        # Use processed context if available for better fact checking
        context = state.processed_context if hasattr(state, 'processed_context') else state.retrieved_context[:1500]
        
        prompt = fact_check_template.format(
            policy_stance=state.policy_stance,
            retrieved_context=context
        )
        
        state.fact_check_result = model.invoke(prompt, temperature=temperature).strip()
        
    except Exception as e:
        logger.error(f"Fact check error: {e}")
        state.fact_check_result = "ERROR"
    
    return state


async def format_response(state: ConversationState) -> ConversationState:
    """Format final response."""
    model = get_model_for_task("format_response")
    temperature = get_temperature_for_task("format_response")
    
    try:
        persona = persona_manager.get_active_persona()
        
        # Format conversation history (limit to last 3 messages)
        history_text = ""
        for message in state.conversation_history[-3:]:
            history_text += f"{message['speaker']}: {message['text']}\n"
        
        # Process fact check if available
        fact_check_info = ""
        if state.fact_check_result and state.fact_check_result not in ["SKIPPED", "ERROR"]:
            if "CONSISTENT" not in state.fact_check_result.upper():
                # Only include if inconsistencies found
                fact_check_info = f"\nFact check: {state.fact_check_result}\n"
                
                # Adjust policy stance if needed
                try:
                    adjust_model = get_model_for_task("adjust_policy")
                    adjustment_prompt = f"""
                    Original stance: {state.policy_stance}
                    Fact check: {state.fact_check_result}
                    Adjust stance to be more factually accurate while maintaining style.
                    """
                    state.policy_stance = adjust_model.invoke(adjustment_prompt, temperature=0.3).strip()
                except Exception as e:
                    logger.warning(f"Could not adjust policy stance: {e}")
        
        prompt = format_response_template.format(
            politician_name=persona["name"],
            politician_party=persona["party"],
            user_input=state.user_input,
            current_topic=state.current_topic,
            should_deflect=state.should_deflect,
            deflection_topic=state.deflection_topic or "",
            policy_stance=state.policy_stance or "",
            speech_patterns=json.dumps(persona["speech_patterns"], indent=2),
            rhetoric_style=json.dumps(persona["rhetorical_style"], indent=2),
            conversation_history=history_text,
            fact_check_info=fact_check_info
        )
        
        response = model.invoke(prompt, temperature=temperature).strip()
        
        state.final_response = response
        state.add_to_history(persona["name"], response)
        
    except Exception as e:
        logger.error(f"Response formatting error: {e}")
        persona = persona_manager.get_active_persona()
        fallback = f"As {persona['name']}, I appreciate your question about {state.current_topic or 'this topic'}."
        state.final_response = fallback
        state.add_to_history(persona["name"], fallback)
    
    return state


async def run_parallel_tasks(state: ConversationState, tasks: List[Callable]) -> ConversationState:
    """Run multiple tasks in parallel with optimal resource allocation."""
    # Create state copies for each task
    states = [state.copy() for _ in tasks]
    
    # Create tasks for parallel execution
    futures = [
        asyncio.create_task(run_with_retry(task, states[i]))
        for i, task in enumerate(tasks)
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*futures, return_exceptions=True)
    
    # Process results and handle any exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Parallel execution error in {tasks[i].__name__}: {result}")
            continue
        
        # Merge results back to main state
        merge_state_updates(state, result, tasks[i].__name__)
    
    return state


def merge_state_updates(target: ConversationState, source: ConversationState, task_name: str) -> None:
    """Merge updates from a parallel task back to the main state."""
    if task_name == "analyze_sentiment":
        target.topic_sentiment = source.topic_sentiment
    elif task_name == "determine_topic":
        target.current_topic = source.current_topic
    elif task_name == "retrieve_context":
        target.retrieved_context = source.retrieved_context
    elif task_name == "process_context":
        target.processed_context = getattr(source, 'processed_context', None)


async def run_with_retry(func: Callable, state: ConversationState) -> ConversationState:
    """Run a function with exponential backoff retry."""
    last_error = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await func(state)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (0.1 * attempt * torch.rand(1).item())
                logger.warning(f"Retry after error in {func.__name__}: {e}. Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
    
    # If we get here, all retries failed
    logger.error(f"All retries failed for {func.__name__}: {last_error}")
    raise last_error


async def run_initial_stage(state: ConversationState) -> ConversationState:
    """First stage with parallel processing of initial analysis."""
    # Always run in parallel for best performance
    return await run_parallel_tasks(
        state,
        [analyze_sentiment, determine_topic, retrieve_context]
    )


async def run_context_stage(state: ConversationState) -> ConversationState:
    """Process retrieved context in parallel with deflection decision."""
    if state.retrieved_context:
        # Process context and decide deflection in parallel
        return await run_parallel_tasks(
            state,
            [process_context, decide_deflection]
        )
    else:
        # Just decide deflection if no context
        return await decide_deflection(state)


def should_fact_check(state: ConversationState) -> str:
    """Determine whether to run fact checking."""
    if state.retrieved_context and not state.should_deflect:
        return "fact_check"
    return "format_response"


def build_graph() -> StateGraph:
    """Build an optimized graph."""
    builder = StateGraph(ConversationState)
    
    # Add optimized workflow nodes
    builder.add_node("initial_analysis", run_initial_stage)
    builder.add_node("context_processing", run_context_stage)
    builder.add_node("generate_policy_stance", generate_policy_stance)
    builder.add_node("fact_check", fact_check)
    builder.add_node("format_response", format_response)
    
    # Connect the workflow with conditional routing
    builder.add_edge("initial_analysis", "context_processing")
    builder.add_edge("context_processing", "generate_policy_stance")
    builder.add_edge("generate_policy_stance", should_fact_check)
    builder.add_edge("fact_check", "format_response")
    builder.add_edge("format_response", END)
    
    # Set entry point
    builder.set_entry_point("initial_analysis")
    
    return builder.compile()


# Create and cache the compiled graph
graph = build_graph()


async def run_conversation(user_input: str) -> str:
    """Run a conversation turn and return the response with performance metrics."""
    state = get_initial_state(user_input)
    
    try:
        # Run the graph with performance tracking
        start_time = time.time()
        
        result = await graph.ainvoke(state)
        
        # Calculate performance metrics
        elapsed = time.time() - start_time
        
        # Log detailed performance
        logger.info(f"Response generated in {elapsed:.2f}s")
        
        # Return the final response
        return result["final_response"]
    except Exception as e:
        logger.error(f"Conversation error: {e}")
        
        # Emergency fallback
        persona = persona_manager.get_active_persona()
        return f"As {persona['name']}, I appreciate your question but am experiencing technical difficulties."
