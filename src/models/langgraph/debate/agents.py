#!/usr/bin/env python3
"""
Debate Agents for the AI Politicians Debate System
==================================================

This module contains the agent implementations for the debate system,
including the moderator, debater, fact-checker, and other role-specific agents.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.config import PoliticianIdentity
from src.models.langgraph.agents.response_agent import generate_response
from src.models.langgraph.agents.context_agent import retrieve_knowledge


def moderate_debate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Moderator agent that guides the debate, enforces rules, and manages transitions.
    
    Args:
        state: The current debate state
        
    Returns:
        Updated state with moderator actions
    """
    result = state.copy()
    current_turn = len(state["turn_history"])
    format_type = state["format"]["format_type"]
    moderator_control = state["format"]["moderator_control"]
    
    # Trace information if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Moderator Agent")
        print("===========================")
        print(f"Turn: {current_turn}")
        print(f"Current Speaker: {state['current_speaker']}")
        print(f"Current Subtopic: {state['current_subtopic']}")
        print("---------------------------")
    
    # Generate moderator message based on current debate state
    moderator_message = ""
    
    # For the first turn, introduce the debate
    if current_turn == 0:
        moderator_message = generate_introduction(state)
    # For transitions, generate transition text
    elif state.get("speaking_queue") and not state.get("interruption_requested"):
        # Rotate speaking queue
        next_speaker = state["speaking_queue"].pop(0) if state["speaking_queue"] else state["participants"][0]
        # Add current speaker to end of queue
        state["speaking_queue"].append(state["current_speaker"])
        # Update current speaker
        result["current_speaker"] = next_speaker
        
        moderator_message = generate_transition(state, next_speaker)
    
    # Add moderator's message to notes if it exists
    if moderator_message:
        result["moderator_notes"].append({
            "turn": current_turn,
            "message": moderator_message,
            "timestamp": datetime.now().isoformat()
        })
    
    return result


def politician_turn(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a politician's speaking turn in the debate.
    
    Args:
        state: The current debate state
        
    Returns:
        Updated state with the politician's response
    """
    result = state.copy()
    current_speaker = state["current_speaker"]
    current_turn = len(state["turn_history"])
    
    # Trace information if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Politician Agent")
        print("============================")
        print(f"Current Speaker: {current_speaker}")
        print(f"Turn: {current_turn}")
        print("----------------------------")
    
    # Check if we should provide knowledge for this turn
    if state["use_rag"]:
        # Get relevant knowledge
        knowledge = retrieve_knowledge_for_debate(
            state["topic"], 
            state["current_subtopic"], 
            current_speaker
        )
        # Update the politician's knowledge
        result["debater_states"][current_speaker]["knowledge"] = knowledge
    
    # Generate the politician's response
    other_participants = [p for p in state["participants"] if p != current_speaker]
    previous_statements = get_recent_statements(state, 3)
    
    response = generate_politician_debate_response(
        identity=current_speaker,
        topic=state["current_subtopic"],
        knowledge=result["debater_states"][current_speaker]["knowledge"],
        previous_statements=previous_statements,
        opponents=other_participants,
        rebuttal_targets=identify_rebuttal_targets(state, current_speaker),
        format_type=state["format"]["format_type"],
        max_length=get_max_response_length(state["format"])
    )
    
    # Record the turn in history
    result["turn_history"].append({
        "turn": current_turn,
        "speaker": current_speaker,
        "statement": response,
        "subtopic": state["current_subtopic"],
        "timestamp": datetime.now().isoformat(),
        "knowledge_used": bool(result["debater_states"][current_speaker]["knowledge"])
    })
    
    # Small chance to trigger an interruption based on debate format
    if (state["format"]["allow_interruptions"] and 
        state["format"]["moderator_control"] != "strict" and
        random.random() < 0.25):  # 25% chance of interruption
        
        # Randomly select an opponent to interrupt
        interrupter = random.choice([p for p in state["participants"] if p != current_speaker])
        result["interruption_requested"] = True
        result["interrupt_by"] = interrupter
    
    return result


def fact_check(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform fact checking on politician statements.
    
    Args:
        state: The current debate state
        
    Returns:
        Updated state with fact check results
    """
    result = state.copy()
    
    if not state["format"]["fact_check_enabled"] or not state["turn_history"]:
        return result
    
    # Get the most recent statement
    latest_turn = state["turn_history"][-1]
    statement = latest_turn["statement"]
    speaker = latest_turn["speaker"]
    
    # Trace information if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Fact Checker Agent")
        print("=============================")
        print(f"Checking statement by: {speaker}")
        print(f"Statement length: {len(statement)} chars")
        print("-----------------------------")
    
    # Extract factual claims for checking
    claims = extract_factual_claims(statement)
    
    if not claims:
        return result
    
    # Check each claim
    checked_claims = []
    for claim in claims:
        # Perform fact checking (simplified here)
        accuracy, corrected_info, sources = check_claim_accuracy(claim)
        
        checked_claims.append({
            "statement": claim,
            "accuracy": accuracy,
            "corrected_info": corrected_info,
            "sources": sources
        })
    
    # Add fact check to the list
    result["fact_checks"].append({
        "turn": len(state["turn_history"]) - 1,
        "speaker": speaker,
        "claims": checked_claims,
        "timestamp": datetime.now().isoformat()
    })
    
    return result


def handle_interruption(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle interruptions during the debate.
    
    Args:
        state: The current debate state
        
    Returns:
        Updated state after handling interruption
    """
    result = state.copy()
    
    if not state["interruption_requested"]:
        return result
    
    interrupter = state.get("interrupt_by", "")
    interrupted = state["current_speaker"]
    
    # Trace information if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Interruption Handler")
        print("===============================")
        print(f"Interrupter: {interrupter}")
        print(f"Interrupted: {interrupted}")
        print("-------------------------------")
    
    # Generate the interruption
    interruption_text = generate_interruption(
        interrupter=interrupter,
        interrupted=interrupted,
        topic=state["current_subtopic"],
        max_length=state["format"]["max_rebuttal_length"]
    )
    
    # Record the interruption
    result["turn_history"].append({
        "turn": len(state["turn_history"]),
        "speaker": interrupter,
        "statement": interruption_text,
        "subtopic": state["current_subtopic"],
        "is_interruption": True,
        "interrupted": interrupted,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update the speaker states
    result["debater_states"][interrupted]["interrupted"] = True
    
    # Reset interruption flags
    result["interruption_requested"] = False
    result["interrupt_by"] = ""
    
    # Set the current speaker to the moderator to handle the aftermath
    # This works through the workflow that always returns to the moderator
    
    return result


def manage_topic(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manage the debate topic and subtopics.
    
    Args:
        state: The current debate state
        
    Returns:
        Updated state with potentially new subtopic
    """
    result = state.copy()
    current_turn = len(state["turn_history"])
    
    # Trace information if enabled
    if state.get("trace", False):
        print("\n🔎 TRACE: Topic Manager")
        print("=========================")
        print(f"Current Topic: {state['topic']}")
        print(f"Current Subtopic: {state['current_subtopic']}")
        print(f"Turn: {current_turn}")
        print("-------------------------")
    
    # Safety check - if we've gone too many turns, just return without changing topic
    # This helps prevent infinite loops
    if current_turn >= 12:
        # Add a final note if we're ending
        result["moderator_notes"].append({
            "turn": current_turn,
            "message": f"We're approaching the end of our debate on {state['topic']}. Please offer your closing statements.",
            "timestamp": datetime.now().isoformat()
        })
        return result
    
    # Identify potential subtopics based on the main topic
    potential_subtopics = generate_subtopics(state["topic"], state["current_subtopic"])
    
    # Check if we should switch to a new subtopic
    if current_turn > 0 and current_turn % 3 == 0:  # Every 3 turns
        # Select a new subtopic
        try:
            new_subtopic = select_next_subtopic(
                potential_subtopics, 
                state["turn_history"],
                state["current_subtopic"]
            )
            
            if new_subtopic != state["current_subtopic"]:
                result["current_subtopic"] = new_subtopic
                
                # Add moderator note about topic change
                result["moderator_notes"].append({
                    "turn": current_turn,
                    "message": f"Let's move on to discuss {new_subtopic}.",
                    "timestamp": datetime.now().isoformat(),
                    "topic_change": True,
                    "old_topic": state["current_subtopic"],
                    "new_topic": new_subtopic
                })
        except Exception as e:
            # If any error occurs, just keep the current subtopic
            print(f"Error managing topics: {e}")
    
    return result


# Helper functions for the agents

def generate_introduction(state: Dict[str, Any]) -> str:
    """Generate the moderator's introduction for the debate."""
    format_type = state["format"]["format_type"]
    participants = ", ".join(state["participants"])
    
    introduction = (
        f"Welcome to today's {format_type} debate on the topic of '{state['topic']}'. "
        f"Participating in this debate are {participants}. "
        f"Each speaker will have {state['format']['time_per_turn']} seconds per turn. "
    )
    
    if state["format"]["allow_interruptions"]:
        introduction += "Interruptions will be allowed during this debate. "
    else:
        introduction += "No interruptions will be permitted. "
    
    if state["format"]["fact_check_enabled"]:
        introduction += "Statements will be fact-checked for accuracy. "
    
    introduction += f"Let's begin with {state['current_speaker']}."
    
    return introduction


def generate_transition(state: Dict[str, Any], next_speaker: str) -> str:
    """Generate a transition between speakers."""
    current_turn = len(state["turn_history"])
    format_type = state["format"]["format_type"]
    
    # Different transitions based on debate format
    if format_type == "town_hall":
        return f"Thank you. Now let's hear from {next_speaker} on this issue."
    elif format_type == "head_to_head":
        return f"Your time is up. {next_speaker}, your response?"
    elif format_type == "panel":
        return f"Let's get {next_speaker}'s perspective on this."
    else:
        return f"Now, {next_speaker}."


def retrieve_knowledge_for_debate(main_topic: str, subtopic: str, identity: str) -> str:
    """Retrieve relevant knowledge for a politician on the debate topic."""
    # This is a wrapper around the existing knowledge retrieval system
    # Simplified implementation for this example
    try:
        # Combine main topic and subtopic for retrieval
        query = f"{main_topic}: {subtopic}"
        return retrieve_knowledge(query, identity)
    except Exception as e:
        print(f"Error retrieving knowledge: {e}")
        return ""


def get_recent_statements(state: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """Get the most recent statements from the debate history."""
    history = state["turn_history"]
    return history[-count:] if len(history) >= count else history


def generate_politician_debate_response(
    identity: str,
    topic: str, 
    knowledge: str,
    previous_statements: List[Dict[str, Any]],
    opponents: List[str],
    rebuttal_targets: List[Tuple[str, str]],
    format_type: str,
    max_length: int = 500
) -> str:
    """Generate a politician's response in the debate context."""
    # This can leverage the existing response generation system
    # Build the prompt with the debate context
    
    # Extract previous statements in a formatted way
    prev_statements_text = ""
    for stmt in previous_statements:
        prev_statements_text += f"{stmt['speaker']}: {stmt['statement']}\n\n"
    
    # Build rebuttal targets
    rebuttal_text = ""
    if rebuttal_targets:
        rebuttal_text = "Consider addressing these points from your opponents:\n"
        for opponent, point in rebuttal_targets:
            rebuttal_text += f"- {opponent} said: {point}\n"
    
    # Build the full context
    context = (
        f"Topic: {topic}\n\n"
        f"Previous statements:\n{prev_statements_text}\n"
        f"Opponents: {', '.join(opponents)}\n"
        f"{rebuttal_text}\n"
        f"Relevant knowledge:\n{knowledge}\n"
    )
    
    # Generate the response using the response agent
    input_state = {
        "user_input": topic,
        "politician_identity": identity,
        "context": context,
        "should_deflect": False
    }
    
    response = generate_response(input_state)
    
    # Truncate if needed
    if len(response) > max_length:
        response = response[:max_length-3] + "..."
    
    return response


def identify_rebuttal_targets(state: Dict[str, Any], current_speaker: str) -> List[Tuple[str, str]]:
    """Identify statements from other politicians that the current speaker might want to rebut."""
    history = state["turn_history"]
    targets = []
    
    # Look through recent history for statements by other speakers
    for turn in reversed(history[-5:] if len(history) > 5 else history):
        if turn["speaker"] != current_speaker:
            # Extract a potential point to rebut (simplified)
            statement = turn["statement"]
            if len(statement) > 100:
                # Just use a portion of the statement for simplicity
                rebuttal_point = statement[:100] + "..."
            else:
                rebuttal_point = statement
            
            targets.append((turn["speaker"], rebuttal_point))
    
    return targets[:2]  # Limit to 2 rebuttal targets


def extract_factual_claims(statement: str) -> List[str]:
    """Extract factual claims from a statement for fact checking."""
    # This is a simplification - in a real system, this would use NLP to identify claims
    # For this example, we'll use a simple approach of splitting into sentences
    # and selecting sentences that look like they contain facts
    
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', statement)
    
    # Look for sentences that might contain factual claims
    factual_indicators = [
        r'\b\d+\s*%', # Percentages
        r'\bin\s+\d{4}\b', # Years
        r'\b(increased|decreased|rose|fell)\b', # Trends
        r'\b(billion|million|trillion)\b', # Large numbers
        r'\b(according to|research shows)\b', # Citations
        r'\b(always|never|all|none)\b', # Absolutes
    ]
    
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if the sentence contains indicators of factual claims
        is_claim = False
        for indicator in factual_indicators:
            if re.search(indicator, sentence, re.IGNORECASE):
                is_claim = True
                break
                
        if is_claim:
            claims.append(sentence)
    
    # Limit to a reasonable number of claims
    return claims[:3]


def check_claim_accuracy(claim: str) -> Tuple[float, Optional[str], List[str]]:
    """
    Check the accuracy of a factual claim.
    
    Returns:
        Tuple of (accuracy score, corrected info if needed, sources)
    """
    # This is a placeholder - would need to be implemented with a real fact-checking system
    # For this example, we'll return random results
    
    import random
    
    # Simulate accuracy score between 0.3 and 1.0
    accuracy = round(random.uniform(0.3, 1.0), 2)
    
    # If accuracy is low, provide a correction
    corrected_info = None
    if accuracy < 0.7:
        corrected_info = f"The actual facts differ: {claim.replace('increased', 'decreased').replace('decreased', 'increased')}"
    
    # Simulate sources
    possible_sources = [
        "Congressional Budget Office report (2022)",
        "Bureau of Labor Statistics data (2023)",
        "Department of Health study (2021)",
        "New York Times analysis (2022)",
        "Wall Street Journal investigation (2023)",
        "PolitiFact fact check (2023)",
        "FactCheck.org verification (2022)"
    ]
    
    # Pick 1-3 random sources
    num_sources = random.randint(1, 3)
    sources = random.sample(possible_sources, num_sources)
    
    return accuracy, corrected_info, sources


def generate_interruption(interrupter: str, interrupted: str, topic: str, max_length: int = 150) -> str:
    """Generate an interruption from one politician to another."""
    # Simplified implementation - this would need to be more sophisticated in a real system
    
    interruption_templates = [
        "That's simply not true! {interrupted} is misleading the audience about {topic}.",
        "I have to interject here. What {interrupted} just said about {topic} is completely wrong.",
        "If I may interrupt - the facts on {topic} are being distorted here.",
        "The American people deserve the truth about {topic}, not these talking points.",
        "Excuse me, but I can't let that statement about {topic} go unchallenged.",
        "That's a mischaracterization of my position on {topic}.",
        "Point of order! Those claims about {topic} are simply not accurate."
    ]
    
    import random
    template = random.choice(interruption_templates)
    interruption = template.format(interrupted=interrupted, topic=topic)
    
    # Additional custom text
    input_state = {
        "user_input": topic,
        "politician_identity": interrupter,
        "context": f"You're interrupting {interrupted} who was speaking about {topic}.",
        "should_deflect": False
    }
    
    additional_text = generate_response(input_state)[:100]
    full_interruption = f"{interruption} {additional_text}"
    
    # Ensure it's not too long
    if len(full_interruption) > max_length:
        full_interruption = full_interruption[:max_length-3] + "..."
    
    return full_interruption


def generate_subtopics(main_topic: str, current_subtopic: str) -> List[str]:
    """Generate a list of potential subtopics for the debate based on the main topic."""
    # Hard-coded subtopics for common debate topics
    subtopics_by_topic = {
        "Climate Change": [
            "Renewable Energy Implementation",
            "Carbon Taxation",
            "Paris Climate Agreement",
            "Green New Deal",
            "Climate Change Impacts on Agriculture",
            "Nuclear Power in the Green Transition"
        ],
        "Economy": [
            "Inflation and Consumer Prices",
            "Job Creation Policies",
            "Tax Reform",
            "Government Spending",
            "Trade Policies and Tariffs",
            "Small Business Support"
        ],
        "Healthcare": [
            "Universal Healthcare",
            "Prescription Drug Prices",
            "Medicare Expansion",
            "Healthcare for Veterans",
            "Mental Health Services",
            "Rural Healthcare Access"
        ],
        "Immigration": [
            "Border Security",
            "Path to Citizenship",
            "Refugee Policy",
            "DACA and Dreamers",
            "Skilled Immigration Reform",
            "Immigration Court System"
        ],
        "Foreign Policy": [
            "Relations with China",
            "Russia and Eastern Europe",
            "Middle East Strategy",
            "NATO Alliances",
            "Trade Agreements",
            "Foreign Aid Programs"
        ]
    }
    
    # Default subtopics if main topic not found
    default_subtopics = [
        f"{main_topic} - Economic Impact",
        f"{main_topic} - Social Implications",
        f"{main_topic} - Historical Context",
        f"{main_topic} - Future Outlook",
        f"{main_topic} - International Perspective",
        f"{main_topic} - Policy Reform"
    ]
    
    # Return relevant subtopics or defaults
    return subtopics_by_topic.get(main_topic, default_subtopics)


def select_next_subtopic(potential_subtopics: List[str], history: List[Dict[str, Any]], current_subtopic: str) -> str:
    """Select the next subtopic for the debate."""
    import random
    
    # Extract subtopics already covered
    covered_subtopics = set()
    for turn in history:
        if "subtopic" in turn:
            covered_subtopics.add(turn["subtopic"])
    
    # Filter out the current subtopic and previously covered ones
    available_subtopics = [
        topic for topic in potential_subtopics 
        if topic != current_subtopic and topic not in covered_subtopics
    ]
    
    # If there are uncovered subtopics, choose one randomly
    if available_subtopics:
        return random.choice(available_subtopics)
    
    # If all subtopics have been covered, just return a different one from current
    other_subtopics = [topic for topic in potential_subtopics if topic != current_subtopic]
    if other_subtopics:
        return random.choice(other_subtopics)
    
    # Fallback - keep the current subtopic
    return current_subtopic


def get_max_response_length(format_config: Dict[str, Any]) -> int:
    """Determine maximum response length based on debate format."""
    format_type = format_config["format_type"]
    
    # Different formats have different expected response lengths
    if format_type == "town_hall":
        return 400  # Longer responses for town hall format
    elif format_type == "head_to_head":
        return 300  # Medium-length responses for head-to-head
    elif format_type == "panel":
        return 250  # Shorter responses for panel discussions
    else:
        return 350  # Default length


def generate_response(state: Dict[str, Any]) -> str:
    """Generate a response from a politician based on their identity and context."""
    try:
        # This is a wrapper around the existing response generation system
        from src.models.langgraph.agents.response_agent import generate_response
        response_data = generate_response(state)
        return response_data.get("response", "I don't have a specific response to that issue.")
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm considering my position on this issue." 