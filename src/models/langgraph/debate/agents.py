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
    
    # Count previous fact checks per politician to ensure balance
    fact_check_counts = {}
    for check in result.get("fact_checks", []):
        check_speaker = check.get("speaker", "unknown")
        fact_check_counts[check_speaker] = fact_check_counts.get(check_speaker, 0) + 1
    
    # Get list of all participants
    all_participants = state.get("participants", [])
    
    # Calculate fact check imbalance
    speaker_count = fact_check_counts.get(speaker, 0)
    other_checks = sum(fact_check_counts.get(p, 0) for p in all_participants if p != speaker)
    
    # Calculate an adjustment probability for fact-checking based on balance
    # If this speaker has been checked more, reduce probability; if less, increase it
    base_probability = 0.5  # Reduced base probability of fact-checking from 0.8 to 0.5
    if len(all_participants) > 1:
        # Normalize: how many more/fewer checks has this speaker had
        check_ratio = (speaker_count / (sum(fact_check_counts.values()) or 1)) 
        # Adjust probability based on imbalance - less checked speakers more likely to be checked
        adjustment = 1.0 - check_ratio * 2  # Adjustment factor (-1 to +1)
        check_probability = min(1.0, max(0.3, base_probability + adjustment * 0.3))
    else:
        check_probability = base_probability
    
    # Apply the probability adjustment for fact-checking
    if random.random() > check_probability and speaker_count > 0:
        # Skip fact-checking with this probability for more balanced coverage
        return result
    
    # Extract factual claims for checking
    claims = extract_factual_claims(statement)
    
    # If no claims were detected but we've never checked this speaker before,
    # extract a claim from the statement anyway to ensure initial coverage
    if not claims and speaker_count == 0 and len(statement) > 20:
        # Just take first substantial sentence as a claim
        import re
        sentences = re.split(r'[.!?]+', statement)
        substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        if substantial_sentences:
            claims = [substantial_sentences[0]]
    
    if not claims:
        return result
    
    # Check each claim - limit to at most 2 claims per statement 
    checked_claims = []
    for claim in claims[:2]:  # Limit to 2 claims max per statement
        # Perform fact checking
        accuracy, corrected_info, sources = check_claim_accuracy(claim)
        
        # Determine the rating label based on accuracy
        rating_categories = [
            (0.95, 1.0, "TRUE"),
            (0.80, 0.95, "MOSTLY TRUE"),
            (0.65, 0.80, "PARTIALLY TRUE"),
            (0.45, 0.65, "MIXED"),
            (0.25, 0.45, "PARTIALLY FALSE"),
            (0.10, 0.25, "MOSTLY FALSE"),
            (0.0, 0.10, "FALSE")
        ]
        
        rating_label = "UNKNOWN"
        for min_val, max_val, label in rating_categories:
            if min_val <= accuracy < max_val:
                rating_label = label
                break
        
        claim_result = {
            "statement": claim,
            "accuracy": accuracy,
            "rating": rating_label
        }
        
        if corrected_info:
            claim_result["corrected_info"] = corrected_info
            
        if sources:
            claim_result["sources"] = sources
            
        checked_claims.append(claim_result)
    
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
    import random
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', statement)
    
    # Look for sentences that might contain factual claims
    # Reduced list to focus on stronger factual indicators
    factual_indicators = [
        # Numbers and statistics
        r'\b\d+\s*%', # Percentages
        r'\bin\s+\d{4}\b', # Years
        r'\b(billion|million|trillion)\b', # Large numbers
        
        # Citations and references
        r'\b(according to|research shows|studies show|data shows)\b', # Citations
        
        # Absolutes and strong claims
        r'\b(always|never|all|none|every|no one)\b', # Absolutes
        
        # Contested claims
        r'\b(hoax|fake|lie|truth|fact)\b', # Truth claims
        
        # Assertions about opponent with specifics
        r'\b(Biden|Trump|Republicans|Democrats) (want|wants|tried|supported) to\b', # Specific claims about opponent's actions
        
        # Simple factual patterns only for clear statistical claims
        r'\b(more|less|better|worse) than \d+\b', # Comparative claims with numbers
    ]
    
    claims = []
    seen_patterns = set()  # Track which patterns have been matched
    
    # First pass: extract claims based on patterns
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Skip very short sentences and obvious opinions
        if len(sentence) < 20 or sentence.lower().startswith(("i think", "i believe", "in my opinion")):
            continue
            
        # Check if the sentence contains indicators of factual claims
        matching_patterns = []
        for i, indicator in enumerate(factual_indicators):
            if re.search(indicator, sentence, re.IGNORECASE):
                matching_patterns.append(i)
                
        if matching_patterns:
            claims.append((sentence, matching_patterns))
            for pattern_idx in matching_patterns:
                seen_patterns.add(pattern_idx)
    
    # If we have very few claims, use a second pass with looser criteria - but with low probability
    if len(claims) < 1 and random.random() < 0.3:  # Only 30% chance to add extra claims
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 25:  # Skip empty or short sentences
                continue
                
            # If this sentence isn't already identified as a claim and is substantial
            if not any(claim[0] == sentence for claim in claims) and len(sentence) > 30:
                # Only include if it has indicators of being factual (like numbers or specific references)
                if re.search(r'\b\d+\b', sentence) or re.search(r'\b(report|data|study|analysis)\b', sentence, re.IGNORECASE):
                    claims.append((sentence, [-1]))
    
    # If still no claims but we have sentences and it's the first time checking this speaker,
    # 50% chance to skip fact checking entirely
    if not claims and sentences and random.random() < 0.5:
        return []
    
    # If still no claims but we have sentences, just pick one substantial sentence
    if not claims and sentences:
        substantial_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        if substantial_sentences and random.random() < 0.4:  # 40% chance to use one
            selected = random.choice(substantial_sentences)
            claims.append((selected, [-1]))
    
    # Extract just the text from the claims
    claim_texts = [claim[0] for claim in claims]
    
    # Limit to a reasonable number of claims
    return claim_texts[:2]  # Reduced from 3 to 2


def check_claim_accuracy(claim: str) -> Tuple[float, Optional[str], List[str]]:
    """
    Check the accuracy of a factual claim.
    
    Returns:
        Tuple of (accuracy score, corrected info if needed, sources)
    """
    # This is a placeholder - would need to be implemented with a real fact-checking system
    # For this example, we'll return somewhat realistic results
    
    import random
    
    # Define rating categories with human-readable labels
    rating_categories = [
        (0.95, 1.0, "TRUE"),
        (0.80, 0.95, "MOSTLY TRUE"),
        (0.65, 0.80, "PARTIALLY TRUE"),
        (0.45, 0.65, "MIXED"),
        (0.25, 0.45, "PARTIALLY FALSE"),
        (0.10, 0.25, "MOSTLY FALSE"),
        (0.0, 0.10, "FALSE")
    ]
    
    # Determine if the claim contains obvious political trigger words
    political_words = ['democrat', 'republican', 'liberal', 'conservative', 
                       'biden', 'trump', 'obama', 'hoax', 'fake', 'corrupt',
                       'socialist', 'radical', 'election', 'taxes', 'border']
    
    # Check if claim contains political keywords
    contains_political = any(word in claim.lower() for word in political_words)
    
    # More generous accuracy distribution - biased toward higher truth values
    if contains_political:
        # More polarized but still more favorable distribution for political claims
        base_accuracy = random.choice([
            random.uniform(0.75, 1.0),  # 60% chance
            random.uniform(0.45, 0.75), # 30% chance
            random.uniform(0.2, 0.45)   # 10% chance
        ] * [6, 3, 1])  # Weighted distribution
    else:
        # More balanced distribution for non-political claims, skewed higher
        base_accuracy = random.uniform(0.5, 1.0)
    
    # Determine the exact accuracy score
    accuracy = round(base_accuracy, 2)
    
    # Find the appropriate rating category
    rating_label = "UNKNOWN"
    for min_val, max_val, label in rating_categories:
        if min_val <= accuracy < max_val:
            rating_label = label
            break
    
    # Generate a correction for claims with low accuracy (only below MIXED rating)
    corrected_info = None
    if accuracy < 0.45:  # For anything rated PARTIALLY FALSE or lower
        # Generate different types of corrections based on the claim
        if "never" in claim.lower():
            corrected_info = f"Correction: There have been some documented instances contrary to this claim."
        elif "always" in claim.lower():
            corrected_info = f"Correction: There are some exceptions to this statement."
        elif any(word in claim.lower() for word in ["all", "every", "none"]):
            corrected_info = f"Correction: The claim uses absolutes that don't reflect the nuanced reality."
        elif any(word in claim.lower() for word in ["billion", "million", "trillion"]):
            corrected_info = f"Correction: The figures cited are inaccurate or lack context."
        else:
            corrected_info = f"Correction: The claim contains inaccuracies or lacks important context."
    
    # Simulate sources - choose relevant sources based on the topic
    economic_sources = [
        "Congressional Budget Office report (2022)",
        "Bureau of Labor Statistics data (2023)",
        "Federal Reserve economic analysis (2023)",
        "Treasury Department figures (2022)"
    ]
    
    political_sources = [
        "PolitiFact fact check (2023)",
        "FactCheck.org verification (2022)",
        "Washington Post Fact Checker (2023)",
        "Snopes investigation (2022)"
    ]
    
    media_sources = [
        "New York Times analysis (2022)",
        "Wall Street Journal investigation (2023)",
        "Reuters fact check (2023)",
        "Associated Press verification (2022)"
    ]
    
    health_sources = [
        "Department of Health study (2021)",
        "CDC report (2023)",
        "WHO guidelines (2022)",
        "National Institutes of Health research (2023)"
    ]
    
    # Determine which category of sources to use based on content
    if any(word in claim.lower() for word in ['economy', 'economic', 'unemployment', 'jobs', 'inflation', 'tax']):
        source_pool = economic_sources
    elif any(word in claim.lower() for word in ['health', 'covid', 'vaccine', 'medical', 'doctor', 'hospital']):
        source_pool = health_sources
    elif contains_political:
        source_pool = political_sources
    else:
        # Mix sources from different categories
        source_pool = random.sample(economic_sources, 1) + random.sample(political_sources, 1) + random.sample(media_sources, 1)
    
    # Pick 1-3 relevant sources
    num_sources = random.randint(1, 3)
    sources = random.sample(source_pool, min(num_sources, len(source_pool)))
    
    # Format the result with the human-readable rating
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
        # Import here to avoid circular imports and recursion
        from src.models.langgraph.agents.response_agent import generate_response as gen_resp
        response_data = gen_resp(state)
        if isinstance(response_data, dict):
            return response_data.get("response", "I don't have a specific response to that issue.")
        else:
            return response_data  # If it's already a string
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm considering my position on this issue." 