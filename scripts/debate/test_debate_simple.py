#!/usr/bin/env python3
"""
Simple test script for the AI Politician Debate System
This provides a simplified mock debate without external dependencies
"""
import sys
from pathlib import Path
import random
import time
from datetime import datetime

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(root_dir))


def main():
    """Run a simplified mock debate."""
    topic = "Climate Change"
    participants = ["biden", "trump"]
    
    print("\n" + "="*80)
    print(f"DEBATE: {topic}")
    print(f"PARTICIPANTS: {', '.join(participants)}")
    print("="*80 + "\n")
    
    # Mock debate turns
    print("MODERATOR: Welcome to today's head-to-head debate on the topic of 'Climate Change'. "
          "Participating in this debate are biden, trump. Each speaker will have 60 seconds per turn. "
          "Interruptions will be allowed during this debate. "
          "Statements will be fact-checked for accuracy. Let's begin with biden.\n")
    
    time.sleep(1)
    
    print("BIDEN: Climate change is an existential threat that requires immediate action. "
          "Under my administration, we've made historic investments in clean energy through the "
          "Inflation Reduction Act, committing over $360 billion to address climate change. "
          "We're on track to cut emissions in half by 2030 and reach net-zero by 2050. "
          "We've rejoined the Paris Climate Agreement and are working with our international "
          "partners to hold all nations accountable.\n")
    
    time.sleep(1)
    
    print("[FACT CHECK] Statements by BIDEN:")
    print("  • Claim: \"We're on track to cut emissions in half by 2030\"")
    print("    Rating: PARTIALLY TRUE (0.75)")
    print("    Sources: Congressional Budget Office report (2022), PolitiFact fact check (2023)\n")
    print("  • Claim: \"We've made historic investments in clean energy through the Inflation Reduction Act, committing over $360 billion to address climate change\"")
    print("    Rating: MOSTLY TRUE (0.89)")
    print("    Sources: Department of Health study (2021), Wall Street Journal investigation (2023)\n")
    
    time.sleep(1)
    
    print("MODERATOR: Your time is up. trump, your response?\n")
    
    time.sleep(1)
    
    print("TRUMP: Biden's climate agenda is killing American jobs and crushing our economy. "
          "These radical Green New Deal policies are sending energy prices through the roof, "
          "while China and India continue to build coal plants every week. When I was President, "
          "we had energy independence for the first time, with lower gas prices and more American "
          "energy jobs.\n")
    
    time.sleep(1)
    
    print("[INTERRUPTION] BIDEN: That's simply not true! trump is misleading the audience about climate change. "
          "The clean energy transition is creating millions of new jobs, and our economy has been growing while emissions decline.\n")
    
    time.sleep(1)
    
    print("[TOPIC CHANGE] MODERATOR: Let's move on to discuss renewable energy.\n")
    
    time.sleep(1)
    
    print("BIDEN: Renewable energy is the future, and America should lead it. "
          "During my administration, we've seen record growth in solar and wind deployment. "
          "The cost of renewable energy has plummeted, making it cheaper than fossil fuels in many parts of the country. "
          "We're making historic investments in upgrading our power grid and building electric vehicle charging stations nationwide.\n")
    
    # Print summary
    print("\n" + "="*80)
    print("DEBATE SUMMARY:")
    print("  Topic: Climate Change")
    print("  Participants: biden, trump")
    print("  Turns: 2")
    print("  Interruptions: 1")
    print("  Fact Checks: 2")
    print("  Subtopics Covered: Climate Change, renewable energy")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 