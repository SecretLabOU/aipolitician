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
import logging
import time

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
    
    # Update the debate memory with significant points from previous turns
    if "debate_memory" not in result:
        result["debate_memory"] = {}
    
    # Generate the politician's response
    other_participants = [p for p in state["participants"] if p != current_speaker]
    previous_statements = get_recent_statements(state, 4)  # Increased from 3 to 4 for more context
    
    # Extract important points from recent opponent statements
    opponent_points = extract_key_points_from_opponents(previous_statements, current_speaker)
    
    # Update the debate memory with these points
    if current_speaker not in result["debate_memory"]:
        result["debate_memory"][current_speaker] = {
            "opponents_addressed": set(),
            "topics_addressed": set(),
            "points_responded_to": set(),
            "own_points_made": []
        }
    
    # Track which opponents and topics have been addressed
    for opponent, point_id in opponent_points:
        result["debate_memory"][current_speaker]["opponents_addressed"].add(opponent)
        result["debate_memory"][current_speaker]["points_responded_to"].add(point_id)
    
    # Generate response considering the debate memory
    response = generate_politician_debate_response(
        identity=current_speaker,
        topic=state["current_subtopic"],
        knowledge=result["debater_states"][current_speaker]["knowledge"],
        previous_statements=previous_statements,
        opponents=other_participants,
        rebuttal_targets=identify_rebuttal_targets(state, current_speaker),
        format_type=state["format"]["format_type"],
        max_length=get_max_response_length(state["format"]),
        debate_memory=result["debate_memory"].get(current_speaker, {})
    )
    
    # Extract key points from the speaker's own response
    speaker_points = extract_key_points(response)
    result["debate_memory"][current_speaker]["own_points_made"].extend(speaker_points)
    result["debate_memory"][current_speaker]["topics_addressed"].add(state["current_subtopic"])
    
    # Record the turn in history
    result["turn_history"].append({
        "turn": current_turn,
        "speaker": current_speaker,
        "statement": response,
        "subtopic": state["current_subtopic"],
        "timestamp": datetime.now().isoformat(),
        "knowledge_used": bool(result["debater_states"][current_speaker]["knowledge"]),
        "key_points": speaker_points
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
    Fact check the latest statement from the debate.
    
    This function extracts factual claims from the latest statement,
    fact checks them, and returns the results.
    """
    import logging
    import time
    from typing import Dict, List, Tuple
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fact_checker")
    
    # Initialize the result dictionary
    result = {
        "fact_checks": state.get("fact_checks", []),
        "latest_fact_check": None,
    }
    
    # Try multiple ways to get the latest statement
    latest_statement = None
    speaker = None
    
    # Option 1: Direct latest_statement in state
    if "latest_statement" in state and state["latest_statement"]:
        latest_statement = state["latest_statement"]
        speaker = state.get("current_speaker", "")
    
    # Option 2: Get from turn_history
    elif "turn_history" in state and state["turn_history"]:
        try:
            latest_turn = state["turn_history"][-1]
            latest_statement = latest_turn.get("statement", "")
            speaker = latest_turn.get("speaker", "")
        except (IndexError, KeyError, TypeError):
            pass
    
    # Debug info
    if latest_statement:
        logger.info(f"Checking statement by {speaker}, length: {len(str(latest_statement))} chars")
    else:
        logger.info("No statement found to check")
        return result
    
    # Ensure we have both statement and speaker
    if not latest_statement or not speaker:
        return result
    
    # Track fact check counts per speaker for balanced coverage
    speaker_fact_checks = state.get("speaker_fact_checks", {})
    speaker_count = speaker_fact_checks.get(speaker, 0)
    
    # Extract claims from the statement
    claims = extract_factual_claims(latest_statement)
    logger.info(f"Extracted {len(claims)} potential claims")
    
    # Filter for actually checkable claims
    checkable_claims = [claim for claim in claims if verify_fact_is_checkable(claim)]
    logger.info(f"Found {len(checkable_claims)} checkable claims")
    
    # If no checkable claims, skip fact checking
    if not checkable_claims:
        logger.info(f"No checkable claims found in statement by {speaker}")
        return result
    
    # Get the accuracy of the claims using real fact checking
    accuracy_scores = []
    all_sources = []
    corrections = []
    
    for claim in checkable_claims[:2]:  # Check at most 2 claims
        # Use the real fact checking function
        accuracy, corrected_info, sources = check_claim_accuracy(claim)
        accuracy_scores.append(accuracy)
        all_sources.extend(sources)
        
        if corrected_info:
            corrections.append(corrected_info)
    
    # Calculate the overall accuracy as the average of the individual accuracies
    overall_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    
    # Determine rating label based on accuracy
    rating = "UNKNOWN"
    if overall_accuracy >= 0.9:
        rating = "TRUE"
    elif overall_accuracy >= 0.75:
        rating = "MOSTLY TRUE"
    elif overall_accuracy >= 0.6:
        rating = "PARTIALLY TRUE"
    elif overall_accuracy >= 0.5:
        rating = "MIXED"
    elif overall_accuracy >= 0.35:
        rating = "PARTIALLY FALSE"
    elif overall_accuracy >= 0.2:
        rating = "MOSTLY FALSE"
    else:
        rating = "FALSE"
    
    # Store the fact check results
    fact_check_result = {
        "speaker": speaker,
        "timestamp": time.time(),
        "claims": checkable_claims[:2],
        "accuracy": overall_accuracy,
        "rating": rating,
        "sources": all_sources,
        "corrections": corrections
    }
    
    # Update the state with the fact check results
    result["fact_checks"] = state.get("fact_checks", []) + [fact_check_result]
    result["latest_fact_check"] = fact_check_result
    
    # Update the speaker fact check count
    speaker_fact_checks[speaker] = speaker_count + 1
    result["speaker_fact_checks"] = speaker_fact_checks
    
    # Log the fact check for debugging
    logger.info(f"Fact check completed for {speaker}: Rating={rating}, Accuracy={overall_accuracy:.2f}")
    
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
    # Prevent infinite loops or excessive topic changes
    if current_turn >= 12 or current_turn == 0:
        # Add a final note if we're ending and near max turns
        if current_turn >= 10:
            result["moderator_notes"].append({
                "turn": current_turn,
                "message": f"We're approaching the end of our debate on {state['topic']}. Please offer your closing statements.",
                "timestamp": datetime.now().isoformat()
            })
        return result
    
    # Only change topics at specific intervals (every 4 turns)
    # This prevents the rapid cycling through topics
    if current_turn > 0 and current_turn % 4 == 0:
        # Identify potential subtopics based on the main topic
        potential_subtopics = generate_subtopics(state["topic"], state["current_subtopic"])
        
        # Track which subtopics have been covered in the state
        if "subtopics_covered" not in result:
            result["subtopics_covered"] = [state["current_subtopic"]]
        
        # Select a new subtopic, avoiding the current one and recently used ones
        try:
            covered_subtopics = result.get("subtopics_covered", [])
            
            # Filter subtopics to those not recently covered
            available_subtopics = [
                topic for topic in potential_subtopics 
                if topic != state["current_subtopic"] and topic not in covered_subtopics[-3:]
            ]
            
            # If we have available subtopics, select one
            if available_subtopics:
                new_subtopic = random.choice(available_subtopics)
                
                if new_subtopic != state["current_subtopic"]:
                    result["current_subtopic"] = new_subtopic
                    result["subtopics_covered"].append(new_subtopic)
                    
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


def get_max_response_length(format_config: Dict[str, Any]) -> int:
    """Determine maximum response length based on debate format."""
    format_type = format_config["format_type"]
    
    # Significantly increase response lengths across all formats
    if format_type == "town_hall":
        return 1000  # Increased from 800
    elif format_type == "head_to_head":
        return 900  # Increased from 600
    elif format_type == "panel":
        return 800  # Increased from 500
    else:
        return 950  # Increased from 700


def generate_politician_debate_response(
    identity: str,
    topic: str, 
    knowledge: str,
    previous_statements: List[Dict[str, Any]],
    opponents: List[str],
    rebuttal_targets: List[Tuple[str, str]],
    format_type: str,
    max_length: int = 500,
    debate_memory: Dict = None
) -> str:
    """Generate a politician's response in the debate context."""
    # Process debate memory to avoid repetition and enhance coherence
    debate_memory = debate_memory or {}
    already_addressed = debate_memory.get("points_responded_to", set())
    own_points = debate_memory.get("own_points_made", [])
    
    # Extract previous statements in a formatted way with special emphasis on new points
    prev_statements_text = ""
    for stmt in previous_statements:
        speaker = stmt['speaker']
        statement = stmt['statement']
        is_opponent = speaker != identity
        
        # Mark statements that haven't been addressed yet
        if is_opponent:
            prev_statements_text += f"{speaker} (opponent): {statement}\n\n"
        else:
            prev_statements_text += f"{speaker} (you, earlier): {statement}\n\n"
    
    # Build rebuttal targets with priority to unaddressed points
    rebuttal_text = ""
    if rebuttal_targets:
        filtered_targets = []
        
        # Prioritize targets that haven't been addressed yet
        for opponent, point in rebuttal_targets:
            point_id = f"{opponent}:{hash(point) % 10000}"
            if point_id not in already_addressed:
                filtered_targets.append((opponent, point, True))  # True = needs addressing
            else:
                filtered_targets.append((opponent, point, False))  # False = already addressed
                
        # Sort to put unaddressed points first
        filtered_targets.sort(key=lambda x: not x[2])
        
        # Build the rebuttal text
        if filtered_targets:
            rebuttal_text = "Consider addressing these points from your opponents:\n"
            for opponent, point, unaddressed in filtered_targets[:3]:  # Limit to 3 points
                prefix = "* IMPORTANT TO ADDRESS" if unaddressed else ""
                rebuttal_text += f"- {opponent} said: {point} {prefix}\n"
    
    # Include guidance to avoid repeating points
    continuity_guidance = ""
    if own_points:
        continuity_guidance = "You've previously made these points (avoid direct repetition):\n"
        for i, point in enumerate(own_points[-3:]):  # Last 3 points
            continuity_guidance += f"- {point}\n"
    
    # Build the full context
    context = (
        f"Topic: {topic}\n\n"
        f"You are {identity}, participating in a debate on '{topic}'.\n"
        f"Previous statements in the debate:\n{prev_statements_text}\n"
        f"Opponents: {', '.join(opponents)}\n"
        f"{rebuttal_text}\n"
        f"{continuity_guidance}\n"
        f"Relevant knowledge:\n{knowledge}\n\n"
        f"IMPORTANT: Respond directly to your opponents' points. Build on the conversation "
        f"rather than repeating yourself. Be specific in your references to what others have said. "
        f"Provide a complete, thorough response in your authentic voice."
    )
    
    # Generate the response using the response agent
    input_state = {
        "user_input": topic,
        "politician_identity": identity,
        "context": context,
        "should_deflect": False,
        "max_new_tokens": 1024,  # Increase max tokens to prevent errors
        "max_length": 1536  # Add high max_length to prevent token length errors
    }
    
    response = generate_response(input_state)
    
    # Only truncate if response is significantly over the max length
    if len(response) > max_length + 300:
        # More gently truncate to preserve more content
        truncate_point = max(max_length, int(len(response) * 0.85))
        # Try to truncate at the end of a sentence
        sentence_end = response.rfind(".", 0, truncate_point)
        if sentence_end > max_length * 0.7:  # Only use sentence boundary if it's reasonably far in
            response = response[:sentence_end+1]
        else:
            response = response[:truncate_point-3] + "..."
    
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


def extract_factual_claims(statement):
    """Extract verifiable factual claims from a statement for fact checking."""
    import re
    import random
    
    # Ensure statement is a string - add type checking/conversion
    if not statement:
        return []
    
    # Handle case where statement might be a dictionary or other object
    if not isinstance(statement, str):
        # Try to convert if it has a string representation
        try:
            if hasattr(statement, 'get') and isinstance(statement.get('statement', ''), str):
                statement = statement.get('statement', '')
            else:
                statement = str(statement)
        except Exception:
            # If conversion fails, return empty list
            print("🔎 DEBUG: Could not convert statement to string")
            return []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', statement)
    
    # FACTUAL PATTERN RECOGNITION - broader than before to catch more claims
    verifiable_patterns = [
        # Numbers and statistics (more general patterns)
        (r'\b\d+', "numbers"),  # Any numbers at all
        (r'percent|percentage|\d+%', "percentage_reference"),
        (r'dollars?|\$|costs?|spending|budget|deficit|debt', "financial_reference"),
        (r'(increased|decreased|reduced|grew|fell|rose|dropped)', "trend_reference"),
        
        # Time references (more general)
        (r'years?|months?|weeks?|days?|in \d+|since \d+|before \d+|after \d+', "time_reference"),
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)', "month_reference"),
        (r'(today|yesterday|last year|this year|next year|decade)', "temporal_reference"),
        
        # Actions and policies (more general)
        (r'(signed|vetoed|voted|implemented|established|created|proposed|introduced)', "action_reference"),
        (r'(bill|legislation|act|policy|program|initiative|law|executive order)', "policy_reference"),
        
        # Jobs and economy references
        (r'(jobs?|employment|unemployment|wages?|salaries?|workers?|labor|manufacturing)', "jobs_reference"),
        (r'(econom(y|ic)|industr(y|ial)|business(es)?|companies?|corporations?)', "economy_reference"),
        
        # Crime and security references
        (r'(crime|safety|security|violence|murder|homicide|theft|police|law enforcement)', "crime_reference"),
        
        # Infrastructure references
        (r'(infrastructure|roads?|bridges?|airports?|ports?|buildings?|construction)', "infrastructure_reference"),
        
        # References to states or places (common in political claims)
        (r'(Michigan|Ohio|Pennsylvania|Wisconsin|Florida|Georgia|Arizona|Texas)', "state_reference"),
        (r'(cities?|states?|counties?|regions?|countries?|nations?)', "location_reference"),
        
        # Claims about groups or institutions
        (r'(Americans?|citizens?|voters?|taxpayers?|people|families|communities)', "people_reference"),
        (r'(government|administration|White House|Congress|Senate|House)', "institution_reference")
    ]
    
    # Explicitly filter out pure opinion statements
    opinion_indicators = [
        r'^(I think|I believe|I feel|In my opinion|In my view)',
        r'(is great|is wonderful|is terrible|is awful|is best|is worst)',
        r'^(We need to|We must|We should)',
        r'(make America great again|America is already great)$'
    ]
    
    claims = []
    
    # Process each sentence
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:  # Skip very short sentences
            continue
        
        # Skip pure opinion statements
        is_opinion = False
        for pattern in opinion_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                is_opinion = True
                break
        
        if is_opinion:
            continue
            
        # Check for factual patterns
        matched_patterns = []
        for pattern, pattern_type in verifiable_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                matched_patterns.append(pattern_type)
        
        # If we have matches and the sentence isn't too generic, consider it a factual claim
        if matched_patterns and not sentence.startswith("Let me be clear"):
            # Score based on number of distinct pattern types matched
            unique_patterns = set(matched_patterns)
            verification_score = len(unique_patterns)
            claims.append((sentence, verification_score, list(unique_patterns)))
    
    # Sort claims by verification score (higher = more likely to be factual)
    claims.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top claims (now up to 3 instead of 2)
    claim_texts = [claim[0] for claim in claims[:3]]
    
    return claim_texts


def verify_fact_is_checkable(claim: str) -> bool:
    """Verify that a claim is actually checkable with objective evidence."""
    import re
    
    # Broader set of verification elements - less restrictive than before
    verification_elements = [
        # Numbers and measurements (any numbers at all now count)
        r'\b\d+',
        r'percent|percentage|\d+%|\$',
        
        # Jobs, economy and performance claims
        r'\b(jobs?|employment|unemployment|wages?|econom(y|ic)|industr(y|ial))',
        
        # Crime and security references
        r'\b(crime|safety|security|violence|murder|homicide|theft|police)',
        
        # Infrastructure references
        r'\b(infrastructure|roads?|bridges?|airports?|construction)',
        
        # Administrative actions
        r'\b(signed|vetoed|voted|implemented|established|created|proposed|introduced)',
        
        # Policy references
        r'\b(bill|legislation|act|policy|program|initiative|law|executive order)',
        
        # Temporal references (any time indicators)
        r'\b(years?|months?|weeks?|days?|since|before|after|during|when)',
        r'\b(today|yesterday|last year|this year|next year|decade)',
        
        # Comparison claims
        r'\b(increased|decreased|reduced|grew|fell|rose|dropped|higher|lower|more|less)',
        
        # State or region references
        r'\b(Michigan|Ohio|Pennsylvania|Wisconsin|Florida|Georgia|Arizona|Texas)',
        r'\b(cities?|states?|counties?|regions?|countries?)'
    ]
    
    # Exclude subjective claims about greatness, values, etc.
    exclusion_patterns = [
        r'^America is (already |truly |really |)(great|strong|wonderful|exceptional)',
        r'^I (think|believe|feel) ',
        r'(should|must|need to|ought to) be',
        r'^(We|People) deserve'
    ]
    
    # Check for exclusions first
    for pattern in exclusion_patterns:
        if re.search(pattern, claim, re.IGNORECASE):
            return False
    
    # Check if the claim contains any verification elements
    for pattern in verification_elements:
        if re.search(pattern, claim, re.IGNORECASE):
            return True
            
    # Default to not checkable if no verification elements found
    return False


def check_claim_accuracy(claim: str) -> Tuple[float, Optional[str], List[Dict[str, str]]]:
    """
    Check the accuracy of a factual claim using external fact-checking services.
    
    This function connects to real fact-checking APIs to verify claims.
    Uses API keys sequentially to optimize usage of free tiers.
    
    Args:
        claim: The claim to verify
        
    Returns:
        Tuple of (accuracy score, corrected info if needed, sources)
    """
    import requests
    import json
    import os
    import time
    from typing import Dict, List, Optional, Tuple, Any
    import logging
    from datetime import datetime, timedelta
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fact_checker")
    
    # Load environment variables from .env file if dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed, using existing environment variables")
    
    # Initialize default return values
    accuracy = 0.5  # Default to neutral/unknown
    corrected_info = None
    sources = []
    
    # Get API usage state from environment or initialize if not present
    api_state = get_api_usage_state()
    
    try:
        # Try APIs in sequence, using the first available one that hasn't hit rate limits
        
        # 1. Try Google Fact Check API if not rate limited
        google_api_key = os.environ.get('GOOGLE_FACT_CHECK_API_KEY')
        if google_api_key and not is_api_rate_limited(api_state, "google"):
            logger.info("Trying Google Fact Check API")
            try:
                google_results = check_with_google_fact_check(claim, google_api_key)
                if google_results:
                    increment_api_usage(api_state, "google")
                    save_api_usage_state(api_state)
                    return google_results
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower() or e.args[0] in [429, 403]:
                    mark_api_rate_limited(api_state, "google")
                    logger.warning(f"Google Fact Check API rate limited: {str(e)}")
                else:
                    logger.error(f"Error with Google Fact Check API: {str(e)}")
        
        # 2. Try ClaimBuster API if not rate limited
        claimbuster_api_key = os.environ.get('CLAIMBUSTER_API_KEY')
        if claimbuster_api_key and not is_api_rate_limited(api_state, "claimbuster"):
            logger.info("Trying ClaimBuster API")
            try:
                claimbuster_results = check_with_claimbuster(claim, claimbuster_api_key)
                if claimbuster_results:
                    increment_api_usage(api_state, "claimbuster")
                    save_api_usage_state(api_state)
                    return claimbuster_results
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower() or e.args[0] in [429, 403]:
                    mark_api_rate_limited(api_state, "claimbuster")
                    logger.warning(f"ClaimBuster API rate limited: {str(e)}")
                else:
                    logger.error(f"Error with ClaimBuster API: {str(e)}")
        
        # 3. Try Serper API if not rate limited
        serper_api_key = os.environ.get('SERPER_API_KEY')
        if serper_api_key and not is_api_rate_limited(api_state, "serper"):
            logger.info("Trying Serper API")
            try:
                serper_results = search_with_serper_and_analyze(claim, serper_api_key)
                if serper_results:
                    increment_api_usage(api_state, "serper")
                    save_api_usage_state(api_state)
                    return serper_results
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower() or e.args[0] in [429, 403]:
                    mark_api_rate_limited(api_state, "serper")
                    logger.warning(f"Serper API rate limited: {str(e)}")
                else:
                    logger.error(f"Error with Serper API: {str(e)}")
        
        # 4. Try News API if not rate limited
        news_api_key = os.environ.get('NEWS_API_KEY')
        if news_api_key and not is_api_rate_limited(api_state, "newsapi"):
            logger.info("Trying News API")
            try:
                news_results = search_with_newsapi_and_analyze(claim, news_api_key)
                if news_results:
                    increment_api_usage(api_state, "newsapi")
                    save_api_usage_state(api_state)
                    return news_results
            except Exception as e:
                if "quota" in str(e).lower() or "rate limit" in str(e).lower() or e.args[0] in [429, 403]:
                    mark_api_rate_limited(api_state, "newsapi")
                    logger.warning(f"News API rate limited: {str(e)}")
                else:
                    logger.error(f"Error with News API: {str(e)}")
        
        # 5. Fallback to web search verification without API keys as last resort
        logger.info("All APIs exhausted or unavailable, using basic web search fallback")
        return basic_web_search_verification(claim)
    
    except Exception as e:
        logger.error(f"Error in fact checking: {str(e)}")
        # Return a neutral result if all fact-checking methods fail
        return 0.5, "Unable to verify this claim due to technical issues.", [
            {"title": "Fact-checking service unavailable", "url": ""}
        ]

def get_api_usage_state() -> Dict[str, Any]:
    """
    Get the current API usage state, tracking rate limits and usage counts.
    Stored in a temporary file to persist between runs.
    """
    import os
    import json
    from datetime import datetime, timedelta
    
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_usage_state.json")
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                
            # Convert stored date strings back to datetime objects
            for api in state:
                if "rate_limited_until" in state[api] and state[api]["rate_limited_until"]:
                    state[api]["rate_limited_until"] = datetime.fromisoformat(state[api]["rate_limited_until"])
                    
                    # Reset rate limit if the time has passed
                    if datetime.now() > state[api]["rate_limited_until"]:
                        state[api]["rate_limited_until"] = None
                
                # Reset daily count if it's a new day
                if "daily_reset_at" in state[api] and state[api]["daily_reset_at"]:
                    last_reset = datetime.fromisoformat(state[api]["daily_reset_at"])
                    if datetime.now().date() > last_reset.date():
                        state[api]["daily_count"] = 0
                        state[api]["daily_reset_at"] = datetime.now().isoformat()
                
            return state
        except Exception as e:
            print(f"Error loading API state: {e}")
    
    # Initialize default state if file doesn't exist or has errors
    return {
        "google": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": None, "daily_reset_at": datetime.now().isoformat()},
        "claimbuster": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": None, "daily_reset_at": datetime.now().isoformat()},
        "serper": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": None, "daily_reset_at": datetime.now().isoformat()},
        "newsapi": {"daily_count": 0, "monthly_count": 0, "rate_limited_until": None, "daily_reset_at": datetime.now().isoformat()}
    }

def save_api_usage_state(state: Dict[str, Any]) -> None:
    """Save the API usage state to a file"""
    import os
    import json
    
    state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_usage_state.json")
    
    # Convert datetime objects to strings for JSON serialization
    serializable_state = {}
    for api, api_state in state.items():
        serializable_state[api] = api_state.copy()
        if "rate_limited_until" in api_state and api_state["rate_limited_until"]:
            if isinstance(api_state["rate_limited_until"], datetime):
                serializable_state[api]["rate_limited_until"] = api_state["rate_limited_until"].isoformat()
    
    try:
        with open(state_file, 'w') as f:
            json.dump(serializable_state, f)
    except Exception as e:
        print(f"Error saving API state: {e}")

def is_api_rate_limited(state: Dict[str, Any], api_name: str) -> bool:
    """Check if an API is currently rate limited"""
    if api_name not in state:
        return False
    
    api_state = state[api_name]
    
    # Check if explicitly rate limited
    if "rate_limited_until" in api_state and api_state["rate_limited_until"]:
        if datetime.now() < api_state["rate_limited_until"]:
            return True
    
    # Check daily limits
    if api_name == "google" and api_state.get("daily_count", 0) >= 100:  # Google typically has 100 requests/day free
        return True
    elif api_name == "claimbuster" and api_state.get("daily_count", 0) >= 50:  # ClaimBuster limit varies
        return True
    elif api_name == "serper" and api_state.get("monthly_count", 0) >= 100:  # Serper has 100 free searches/month
        return True
    elif api_name == "newsapi" and api_state.get("daily_count", 0) >= 100:  # NewsAPI has 100 requests/day free
        return True
    
    return False

def mark_api_rate_limited(state: Dict[str, Any], api_name: str) -> None:
    """Mark an API as rate limited for a period of time"""
    if api_name not in state:
        state[api_name] = {}
    
    # Set rate limited for different durations based on the API
    if api_name == "google":
        state[api_name]["rate_limited_until"] = datetime.now() + timedelta(hours=24)
    elif api_name == "claimbuster":
        state[api_name]["rate_limited_until"] = datetime.now() + timedelta(hours=24)
    elif api_name == "serper":
        state[api_name]["rate_limited_until"] = datetime.now() + timedelta(hours=24)
    elif api_name == "newsapi":
        state[api_name]["rate_limited_until"] = datetime.now() + timedelta(hours=24)

def increment_api_usage(state: Dict[str, Any], api_name: str) -> None:
    """Increment the usage count for an API"""
    if api_name not in state:
        state[api_name] = {"daily_count": 0, "monthly_count": 0, "daily_reset_at": datetime.now().isoformat()}
    
    state[api_name]["daily_count"] = state[api_name].get("daily_count", 0) + 1
    state[api_name]["monthly_count"] = state[api_name].get("monthly_count", 0) + 1

def search_with_serper_and_analyze(query: str, api_key: str) -> Tuple[float, Optional[str], List[Dict[str, str]]]:
    """Search with Serper API and analyze results to estimate claim accuracy"""
    search_results = search_with_serper(f"fact check {query}", api_key)
    if not search_results:
        return None
    
    return analyze_search_results(query, search_results)

def search_with_newsapi_and_analyze(query: str, api_key: str) -> Tuple[float, Optional[str], List[Dict[str, str]]]:
    """Search with NewsAPI and analyze results to estimate claim accuracy"""
    search_results = search_with_newsapi(f"fact check {query}", api_key)
    if not search_results:
        return None
    
    return analyze_search_results(query, search_results)

def analyze_search_results(claim: str, search_results: List[Dict]) -> Tuple[float, Optional[str], List[Dict[str, str]]]:
    """Analyze search results to estimate claim accuracy"""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Try to download nltk data if not already present
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    
    # Initialize SentimentIntensityAnalyzer for basic sentiment analysis
    sia = SentimentIntensityAnalyzer()
    
    sources = []
    source_urls = []
    sentiment_scores = []
    
    # Process search results to extract sources and analyze sentiment
    for result in search_results[:5]:  # Limit to top 5 results
        title = result.get('title', '')
        url = result.get('link', result.get('url', ''))
        
        # Skip if we already have this URL or if it's not a valid URL
        if not url or url in source_urls:
            continue
            
        source_urls.append(url)
        
        # Add to sources list
        source = {
            "title": title,
            "url": url
        }
        
        # Analyze sentiment of the title to gauge if it supports or refutes the claim
        sentiment = sia.polarity_scores(title)
        sentiment_scores.append(sentiment['compound'])
        
        # Add to sources
        sources.append(source)
    
    # Estimate accuracy based on sentiment analysis
    # This is a rough heuristic and not a true fact-check
    if sentiment_scores:
        # Average sentiment score
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Map to an accuracy score (0.0-1.0)
        # Positive sentiment generally correlates with supporting the claim
        accuracy = (avg_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Generate corrected info for likely false claims
        corrected_info = None
        if accuracy < 0.4:
            corrected_info = "Based on available sources, this claim may be disputed."
            
        return accuracy, corrected_info, sources
    
    # If no sentiment scores, return default
    return 0.5, None, sources

def basic_web_search_verification(claim: str) -> Tuple[float, Optional[str], List[Dict[str, str]]]:
    """
    Basic fallback method when all APIs are exhausted.
    This just returns a neutral result with a message.
    """
    return 0.5, "Unable to verify with external services. Please check manually.", [
        {"title": "No API keys available or all APIs rate limited", "url": ""}
    ]

def check_with_google_fact_check(claim: str, api_key: str) -> Optional[Tuple[float, Optional[str], List[Dict[str, str]]]]:
    """Check a claim using Google's Fact Check API"""
    import requests
    import json
    
    base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "key": api_key,
        "query": claim,
        "languageCode": "en"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        return None
        
    data = response.json()
    
    # Check if there are any results
    if "claims" not in data or not data["claims"]:
        return None
    
    sources = []
    ratings = []
    
    for fc in data["claims"]:
        # Get the claim review data
        for review in fc.get("claimReview", []):
            publisher = review.get("publisher", {}).get("name", "Unknown source")
            url = review.get("url", "")
            title = review.get("title", "Fact check")
            
            # Extract rating if available
            rating_value = review.get("textualRating", "")
            
            # Add to sources
            sources.append({
                "title": f"{publisher}: {title}",
                "url": url
            })
            
            # Try to map textual rating to numerical score
            score = map_rating_to_score(rating_value)
            if score is not None:
                ratings.append(score)
    
    # Calculate average accuracy if ratings were found
    if ratings:
        accuracy = sum(ratings) / len(ratings)
        
        # Generate corrected info based on accuracy
        corrected_info = None
        if accuracy < 0.5:
            corrected_info = "This claim appears to be disputed by fact-checkers."
            
        return accuracy, corrected_info, sources
    
    return None

def check_with_claimbuster(claim: str, api_key: str) -> Optional[Tuple[float, Optional[str], List[Dict[str, str]]]]:
    """Check a claim using ClaimBuster API"""
    import requests
    import json
    
    # First, assess if the statement is check-worthy
    check_worthy_url = "https://idir.uta.edu/claimbuster/api/v2/score/text/"
    headers = {"x-api-key": api_key}
    
    response = requests.get(check_worthy_url + claim, headers=headers)
    
    if response.status_code != 200:
        return None
        
    score_data = response.json()
    checkworthy_score = score_data.get("results", {}).get("score", 0)
    
    # Only proceed if the claim is worth checking (score > 0.4)
    if checkworthy_score < 0.4:
        return None
    
    # Now search for previously fact-checked similar claims
    search_url = "https://idir.uta.edu/claimbuster/api/v2/factcheck/search/"
    response = requests.get(search_url + claim, headers=headers)
    
    if response.status_code != 200:
        return None
        
    search_data = response.json()
    fact_checks = search_data.get("results", [])
    
    if not fact_checks:
        return None
    
    # Process the fact checks
    sources = []
    ratings = []
    
    for check in fact_checks[:3]:  # Limit to top 3 matches
        source = check.get("source", "Unknown")
        title = check.get("title", "Fact check")
        url = check.get("url", "")
        conclusion = check.get("conclusion", "")
        
        # Add to sources
        sources.append({
            "title": f"{source}: {title}",
            "url": url
        })
        
        # Try to map conclusion to numerical score
        score = map_rating_to_score(conclusion)
        if score is not None:
            ratings.append(score)
    
    # Calculate average accuracy if ratings were found
    if ratings:
        accuracy = sum(ratings) / len(ratings)
        
        # Generate corrected info based on accuracy
        corrected_info = None
        if accuracy < 0.5:
            corrected_info = "This claim appears to be disputed by fact-checkers."
            
        return accuracy, corrected_info, sources
    
    return None

def search_with_serper(query: str, api_key: str) -> List[Dict]:
    """Search the web using Serper API"""
    import requests
    import json
    
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return []
        
    data = response.json()
    organic_results = data.get("organic_results", [])
    
    return organic_results

def search_with_newsapi(query: str, api_key: str) -> List[Dict]:
    """Search news articles using NewsAPI"""
    import requests
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "relevancy"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        return []
        
    data = response.json()
    articles = data.get("articles", [])
    
    # Format to match expected structure
    return [{"title": a["title"], "url": a["url"], "link": a["url"]} for a in articles]

def map_rating_to_score(rating: str) -> Optional[float]:
    """Map textual ratings to numerical scores between 0 and 1"""
    rating = rating.lower()
    
    # True ratings
    if any(term in rating for term in ["true", "accurate", "correct", "verified", "confirmed"]):
        return 0.9
    
    # Mostly true ratings
    if any(term in rating for term in ["mostly true", "largely true", "generally true", "mostly correct"]):
        return 0.75
    
    # Mixed/Partial ratings
    if any(term in rating for term in ["mixture", "mixed", "partly true", "partially true", "half true"]):
        return 0.5
    
    # Mostly false ratings
    if any(term in rating for term in ["mostly false", "largely false", "generally false"]):
        return 0.25
    
    # False ratings
    if any(term in rating for term in ["false", "inaccurate", "incorrect", "wrong", "pants on fire"]):
        return 0.1
    
    # If we can't determine the rating, return None
    return None

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


def extract_key_points_from_opponents(previous_statements: List[Dict[str, Any]], current_speaker: str) -> List[Tuple[str, str]]:
    """Extract key points from opponents' statements that should be addressed."""
    opponent_points = []
    
    # Look at recent statements by opponents
    for stmt in reversed(previous_statements):
        if stmt["speaker"] != current_speaker:
            # Extract key points from this opponent statement
            statement = stmt["statement"]
            speaker = stmt["speaker"]
            
            # If the statement has pre-extracted key points, use those
            if "key_points" in stmt:
                for point in stmt["key_points"]:
                    point_id = f"{speaker}:{hash(point) % 10000}"
                    opponent_points.append((speaker, point_id))
            else:
                # Otherwise extract the key points now
                points = extract_key_points(statement)
                for point in points:
                    point_id = f"{speaker}:{hash(point) % 10000}"
                    opponent_points.append((speaker, point_id))
    
    return opponent_points


def extract_key_points(statement: str) -> List[str]:
    """Extract key points from a statement for the debate memory system."""
    import re
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', statement)
    key_points = []
    
    # Identify sentences that look like key points
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:
            continue
            
        # Check for indicators of a key point
        is_key_point = False
        indicators = [
            # Policy positions
            r'\b(support|oppose|will|would|should|must|need to|plan to)\b',
            # Strong assertions
            r'\b(absolutely|certainly|definitely|clearly|always|never)\b',
            # Comparative claims
            r'\b(more|less|better|worse|higher|lower|stronger|weaker)\b',
            # Issue statements
            r'\b(problem|solution|issue|crisis|challenge|opportunity)\b',
            # Facts and figures
            r'\b\d+\s*%',
            r'\bin\s+\d{4}\b',
            r'\b(million|billion|trillion)\b',
        ]
        
        for indicator in indicators:
            if re.search(indicator, sentence, re.IGNORECASE):
                is_key_point = True
                break
                
        if is_key_point:
            # Limit length of key points for memory efficiency
            if len(sentence) > 100:
                sentence = sentence[:100] + "..."
            key_points.append(sentence)
    
    # Limit to 3 key points per statement
    return key_points[:3]


def generate_response(state: Dict[str, Any]) -> str:
    """Generate a response from a politician based on their identity and context."""
    try:
        # Import here to avoid circular imports and recursion
        from src.models.langgraph.agents.response_agent import generate_response as gen_resp
        
        # Add max_new_tokens parameter to avoid token length errors
        input_state = state.copy()
        input_state["max_new_tokens"] = 1024  # Significantly increase max tokens to prevent errors
        
        # Add high max_length to prevent token length errors
        input_state["max_length"] = 1536  # Ensure this is higher than any likely input length
        
        response_data = gen_resp(input_state)
        if isinstance(response_data, dict):
            return response_data.get("response", "I don't have a specific response to that issue.")
        else:
            return response_data  # If it's already a string
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm considering my position on this issue." 