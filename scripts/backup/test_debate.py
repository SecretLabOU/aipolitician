#!/usr/bin/env python3
"""
Test script for the AI Politician Debate System
This simplified version doesn't rely on external dependencies like RAG
"""
import sys
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(root_dir))

from src.models.langgraph.debate.workflow import DebateFormat, DebateInput


def simulate_debate(topic: str, participants: List[str], format_type: str = "head_to_head") -> Dict[str, Any]:
    """
    Run a simulated debate without using the full LangGraph workflow
    """
    print(f"Starting simulated debate on: {topic}")
    print(f"Participants: {', '.join(participants)}")
    print(f"Format: {format_type}")
    print("-" * 60)
    
    # Create format config
    format_config = DebateFormat(
        format_type=format_type,
        time_per_turn=60,
        allow_interruptions=True,
        fact_check_enabled=True,
        max_rebuttal_length=250,
        moderator_control="moderate"
    )
    
    # Simulate debate rounds
    moderator_notes = []
    turn_history = []
    fact_checks = []
    
    # Add introduction
    moderator_notes.append({
        "turn": 0,
        "message": f"Welcome to today's {format_type} debate on the topic of '{topic}'. "
                   f"Participating in this debate are {', '.join(participants)}. "
                   f"Each speaker will have 60 seconds per turn. "
                   f"Interruptions will be allowed during this debate. "
                   f"Statements will be fact-checked for accuracy. "
                   f"Let's begin with {participants[0]}.",
        "timestamp": datetime.now().isoformat()
    })
    
    # Simulate turns
    current_speaker_idx = 0
    current_turn = 0
    current_subtopic = topic
    subtopics = generate_subtopics(topic)
    
    for i in range(4):  # Simulate 4 turns
        current_speaker = participants[current_speaker_idx]
        
        # Generate response for current speaker
        response = generate_mock_response(current_speaker, current_subtopic)
        
        # Record the turn
        turn_history.append({
            "turn": current_turn,
            "speaker": current_speaker,
            "statement": response,
            "subtopic": current_subtopic,
            "timestamp": datetime.now().isoformat(),
            "knowledge_used": True
        })
        
        # Generate fact check
        claims = extract_factual_claims(response)
        if claims and format_config.fact_check_enabled:
            checked_claims = []
            for claim in claims:
                accuracy, corrected_info, sources = check_claim_accuracy(claim)
                checked_claims.append({
                    "statement": claim,
                    "accuracy": accuracy,
                    "corrected_info": corrected_info,
                    "sources": sources
                })
            
            fact_checks.append({
                "turn": current_turn,
                "speaker": current_speaker,
                "claims": checked_claims,
                "timestamp": datetime.now().isoformat()
            })
        
        # Check for interruption
        if format_config.allow_interruptions and random.random() < 0.3 and len(participants) > 1:
            interrupter_idx = (current_speaker_idx + 1) % len(participants)
            interrupter = participants[interrupter_idx]
            
            interruption_text = generate_interruption(interrupter, current_speaker, current_subtopic)
            
            # Record the interruption
            turn_history.append({
                "turn": current_turn + 0.5,
                "speaker": interrupter,
                "statement": interruption_text,
                "subtopic": current_subtopic,
                "is_interruption": True,
                "interrupted": current_speaker,
                "timestamp": datetime.now().isoformat()
            })
        
        # Change subtopic sometimes
        if (i + 1) % 2 == 0 and subtopics:
            next_subtopic = subtopics.pop(0)
            current_subtopic = next_subtopic
            
            # Add moderator note about topic change
            moderator_notes.append({
                "turn": current_turn + 1,
                "message": f"Let's move on to discuss {next_subtopic}.",
                "timestamp": datetime.now().isoformat(),
                "topic_change": True,
                "old_topic": current_subtopic,
                "new_topic": next_subtopic
            })
        
        # Moderator transition
        current_speaker_idx = (current_speaker_idx + 1) % len(participants)
        next_speaker = participants[current_speaker_idx]
        
        # Add moderator transition note
        moderator_notes.append({
            "turn": current_turn + 1,
            "message": f"Your time is up. {next_speaker}, your response?",
            "timestamp": datetime.now().isoformat()
        })
        
        current_turn += 1
    
    # Compile the debate result
    result = {
        "topic": topic,
        "participants": participants,
        "turn_history": turn_history,
        "fact_checks": fact_checks,
        "moderator_notes": moderator_notes,
        "subtopics_covered": [topic] + [note.get("new_topic", "") for note in moderator_notes if note.get("topic_change")]
    }
    
    return result


def display_debate(debate_result: Dict[str, Any]):
    """Display a debate in a readable format."""
    topic = debate_result["topic"]
    participants = ", ".join(debate_result["participants"])
    
    print(f"\n{'='*80}")
    print(f"DEBATE: {topic}")
    print(f"PARTICIPANTS: {participants}")
    print(f"{'='*80}\n")
    
    # Collect all events (turns, moderator notes, fact checks)
    events = []
    
    # Add turns to events
    for turn in debate_result["turn_history"]:
        events.append({
            "type": "turn",
            "turn": turn["turn"],
            "data": turn,
            "timestamp": turn.get("timestamp", "")
        })
    
    # Add moderator notes to events
    for note in debate_result.get("moderator_notes", []):
        events.append({
            "type": "moderator",
            "turn": note["turn"],
            "data": note,
            "timestamp": note.get("timestamp", "")
        })
    
    # Add fact checks to events
    for check in debate_result.get("fact_checks", []):
        events.append({
            "type": "fact_check",
            "turn": check["turn"],
            "data": check,
            "timestamp": check.get("timestamp", "")
        })
    
    # Sort events by turn number and timestamp
    events.sort(key=lambda e: (e["turn"], e["timestamp"]))
    
    # Display events in order
    for event in events:
        if event["type"] == "turn":
            speaker = event["data"]["speaker"].upper()
            statement = event["data"]["statement"]
            
            # Format based on whether it's an interruption
            if event["data"].get("is_interruption", False):
                print(f"\n[INTERRUPTION] {speaker}: {statement}\n")
            else:
                print(f"\n{speaker}: {statement}\n")
                
        elif event["type"] == "moderator":
            message = event["data"]["message"]
            
            # Special formatting for topic changes
            if event["data"].get("topic_change", False):
                print(f"\n[TOPIC CHANGE] MODERATOR: {message}\n")
            else:
                print(f"\nMODERATOR: {message}\n")
                
        elif event["type"] == "fact_check":
            speaker = event["data"]["speaker"].upper()
            
            output = f"\n[FACT CHECK] Statements by {speaker}:\n"
            
            for claim in event["data"]["claims"]:
                accuracy = claim["accuracy"]
                statement = claim["statement"]
                
                # Color-code accuracy
                if accuracy >= 0.8:
                    accuracy_str = f"MOSTLY TRUE ({accuracy:.2f})"
                elif accuracy >= 0.6:
                    accuracy_str = f"PARTIALLY TRUE ({accuracy:.2f})"
                elif accuracy >= 0.4:
                    accuracy_str = f"PARTIALLY FALSE ({accuracy:.2f})"
                else:
                    accuracy_str = f"MOSTLY FALSE ({accuracy:.2f})"
                
                output += f"  • Claim: \"{statement}\"\n"
                output += f"    Rating: {accuracy_str}\n"
                
                if claim.get("corrected_info"):
                    output += f"    Correction: {claim['corrected_info']}\n"
                
                if claim.get("sources"):
                    output += f"    Sources: {', '.join(claim['sources'])}\n"
                
                output += "\n"
            
            print(output)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"DEBATE SUMMARY:")
    print(f"  Topic: {topic}")
    print(f"  Participants: {participants}")
    print(f"  Turns: {len([t for t in debate_result['turn_history'] if not t.get('is_interruption', False)])}")
    print(f"  Interruptions: {len([t for t in debate_result['turn_history'] if t.get('is_interruption', False)])}")
    print(f"  Fact Checks: {len(debate_result.get('fact_checks', []))}")
    print(f"  Subtopics Covered: {', '.join([s for s in debate_result.get('subtopics_covered', []) if s])}")
    print(f"{'='*80}\n")


# Helper functions

def generate_mock_response(speaker: str, topic: str) -> str:
    """Generate a mock response for a politician."""
    biden_responses = {
        "Climate Change": "Climate change is an existential threat that requires immediate action. Under my administration, we've made historic investments in clean energy through the Inflation Reduction Act, committing over $360 billion to address climate change. We're on track to cut emissions in half by 2030 and reach net-zero by 2050. We've rejoined the Paris Climate Agreement and are working with our international partners to hold all nations accountable.",
        "renewable energy": "Renewable energy is the future, and America should lead it. During my administration, we've seen record growth in solar and wind deployment. The cost of renewable energy has plummeted, making it cheaper than fossil fuels in many parts of the country. We're making historic investments in upgrading our power grid and building electric vehicle charging stations nationwide.",
        "Economy": "When I took office, our economy was in crisis. Now, we've created over 13 million new jobs – more jobs in two years than any president created in a four-year term. Unemployment is at near-historic lows, including record lows for African Americans and Hispanic Americans. Inflation has come down significantly from its peak, and we're rebuilding America's infrastructure through the Bipartisan Infrastructure Law.",
    }
    
    trump_responses = {
        "Climate Change": "Biden's climate agenda is killing American jobs and crushing our economy. These radical Green New Deal policies are sending energy prices through the roof, while China and India continue to build coal plants every week. When I was President, we had energy independence for the first time, with lower gas prices and more American energy jobs.",
        "renewable energy": "Look, I'm all for renewable energy. I'm for solar, I'm for wind, I'm for everything. But it has to be affordable and it can't collapse our energy grid. These windmills kill all the birds and they make a terrible noise. Under Biden, electricity prices are up 30% and gas prices through the roof. We need an all-of-the-above approach.",
        "Economy": "Under my leadership, we had the greatest economy in the history of our country. The stock market hit record after record. We had the lowest unemployment for African Americans, Hispanic Americans, and Asian Americans ever recorded. I cut taxes and regulations like nobody has ever seen, and we were respected around the world again.",
    }
    
    sanders_responses = {
        "Climate Change": "Climate change is not only the existential crisis of our time, it is also an opportunity to create millions of good-paying jobs. We need a Green New Deal that rapidly transitions us to 100% renewable energy. The fossil fuel industry has known about climate change for decades, yet they've spent millions on disinformation while taking billions in subsidies.",
        "renewable energy": "We need to transform our energy system away from fossil fuels to renewable energy. We must recognize that fossil fuel executives have known about climate change for 40 years. Yet they have deliberately prevented action to protect their profits while the planet burns. That is nothing less than criminal activity.",
        "Economy": "We have an economy that is fundamentally broken when the three wealthiest people own more wealth than the bottom half of American society. While working people struggle to put food on the table, billionaires have increased their wealth by over $2 trillion during the pandemic. That level of inequality is not only immoral, it's unsustainable.",
    }
    
    responses = {
        "biden": biden_responses,
        "trump": trump_responses,
        "sanders": sanders_responses
    }
    
    # Get response for the speaker and topic, or default response
    speaker_responses = responses.get(speaker.lower(), {})
    
    # Try to match the exact topic, if not, use a default one
    response = speaker_responses.get(topic, speaker_responses.get("Climate Change", 
                f"As {speaker}, I believe we need serious solutions to address {topic}."))
    
    return response


def generate_interruption(interrupter: str, interrupted: str, topic: str) -> str:
    """Generate an interruption from one politician to another."""
    interruption_templates = [
        "That's simply not true! {interrupted} is misleading the audience about {topic}.",
        "I have to interject here. What {interrupted} just said about {topic} is completely wrong.",
        "If I may interrupt - the facts on {topic} are being distorted here.",
        "The American people deserve the truth about {topic}, not these talking points.",
        "Excuse me, but I can't let that statement about {topic} go unchallenged.",
        "That's a mischaracterization of my position on {topic}.",
        "Point of order! Those claims about {topic} are simply not accurate."
    ]
    
    template = random.choice(interruption_templates)
    interruption = template.format(interrupted=interrupted, topic=topic)
    
    # Short additional text based on interrupter
    if interrupter.lower() == "biden":
        additional_text = "Look, here's the deal - the facts matter."
    elif interrupter.lower() == "trump":
        additional_text = "It's fake news, totally fake. Everyone knows it."
    elif interrupter.lower() == "sanders":
        additional_text = "The American people are tired of the same old establishment lies."
    else:
        additional_text = "We need to set the record straight."
    
    full_interruption = f"{interruption} {additional_text}"
    
    # Ensure it's not too long
    max_length = 200
    if len(full_interruption) > max_length:
        full_interruption = full_interruption[:max_length-3] + "..."
    
    return full_interruption


def extract_factual_claims(statement: str) -> List[str]:
    """Extract factual claims from a statement for fact checking."""
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
    return claims[:2]  # Limiting to 2 for the mock version


def check_claim_accuracy(claim: str) -> Tuple[float, str, List[str]]:
    """Mock function to check the accuracy of a factual claim."""
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


def generate_subtopics(main_topic: str) -> List[str]:
    """Generate subtopics for a main debate topic."""
    topic_areas = {
        "climate change": ["renewable energy", "emissions targets", "paris agreement", "carbon tax"],
        "economy": ["job creation", "inflation", "taxation", "government spending"],
        "healthcare": ["healthcare access", "insurance costs", "prescription drug prices", "medicare for all"],
        "immigration": ["border security", "path to citizenship", "illegal immigration", "refugee policy"],
        "education": ["student loan debt", "free college", "public education funding", "charter schools"],
        "foreign policy": ["international alliances", "military spending", "diplomacy", "trade agreements"]
    }
    
    # Get subtopics for the main topic, or use generic ones
    main_topic_lower = main_topic.lower()
    for topic_key, subtopics in topic_areas.items():
        if topic_key in main_topic_lower:
            return subtopics
    
    # Return generic subtopics if no match
    return ["policy implications", "economic impact", "social considerations", "historical context"]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the AI Politician Debate System")
    parser.add_argument("--topic", type=str, default="Climate Change", 
                      help="Main debate topic")
    parser.add_argument("--participants", type=str, default="biden,trump",
                      help="Comma-separated list of politician identities")
    parser.add_argument("--format", type=str, default="head_to_head",
                      choices=["town_hall", "head_to_head", "panel"],
                      help="Debate format type")
    
    args = parser.parse_args()
    
    # Run the simulated debate
    participants = [p.strip() for p in args.participants.split(",")]
    debate_result = simulate_debate(args.topic, participants, args.format)
    
    # Display the results
    display_debate(debate_result) 