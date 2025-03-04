#!/usr/bin/env python3
"""
Test script for the political RAG database system.
"""
import sys
from db.database import get_database
from db.utils.rag_utils import retrieve_context_for_query

def display_politician_bio(name, politician_id):
    """Display basic biographical information about a politician."""
    print(f"\n===== {name}'s Biography =====")
    bio_db = get_database('biography')
    bio = bio_db.get_complete_biography(politician_id)
    
    # Basic info
    print(f"Name: {bio['politician']['name']}")
    if bio['politician']['birth_date']:
        print(f"Birth Date: {bio['politician']['birth_date']}")
    if bio['politician']['birth_place']:
        print(f"Birth Place: {bio['politician']['birth_place']}")
    
    # Education
    if bio['education']:
        print("\nEducation:")
        for edu in bio['education']:
            degree_info = f"{edu['degree']} in {edu['field_of_study']}" if edu['degree'] and edu['field_of_study'] else edu['degree'] or "Attended"
            years = f"{edu['start_year']} - {edu['end_year'] or 'present'}" if edu['start_year'] else ""
            print(f"- {edu['institution']}: {degree_info} ({years})")
    
    # Career
    if bio['career']:
        print("\nCareer:")
        for career in bio['career']:
            years = f"{career['start_year']} - {career['end_year'] or 'present'}" if career['start_year'] else ""
            print(f"- {career['role']} at {career['organization']} ({years})")
            if career['description']:
                print(f"  {career['description']}")
    
    # Family
    if bio['family']:
        print("\nFamily:")
        for member in bio['family']:
            print(f"- {member['relationship'].capitalize()}: {member['name']}")
            if member['description']:
                print(f"  {member['description']}")

def test_search_query(query, politician):
    """Test searching the databases with a query."""
    print(f"\n===== Query: '{query}' (for {politician}) =====")
    context = retrieve_context_for_query(query, politician)
    print(context)

def main():
    """Main function to run tests."""
    print("Political RAG Database System Test\n")
    
    # Display Trump and Biden bios
    display_politician_bio("Donald Trump", 1)
    display_politician_bio("Joe Biden", 2)
    
    # Test queries
    queries = [
        ("What is Donald Trump's education?", "Donald Trump"),
        ("What positions did Biden hold in government?", "Joe Biden"),
        ("When was Trump born?", "Donald Trump"),
        ("Tell me about Biden's family", "Joe Biden"),
    ]
    
    for query, politician in queries:
        test_search_query(query, politician)

if __name__ == "__main__":
    main()