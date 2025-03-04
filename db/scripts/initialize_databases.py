"""
Initialize all databases in the Political RAG system.

This script creates all the database schemas and populates them with
initial data for Donald Trump and Joe Biden.
"""
import os
import time
from pathlib import Path
import sys

# Add the project root to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from db.database import get_database
from db.config import DATABASE_ROOT

# Ensure the database directory exists
os.makedirs(DATABASE_ROOT, exist_ok=True)


def initialize_databases():
    """Initialize all databases in the system."""
    print(f"Initializing databases in {DATABASE_ROOT}")
    
    # Get all database names (only using implemented databases for now)
    db_names = [
        'biography',
        'policy',
        'voting_record',
        'public_statements',
        # The following databases are planned but not implemented yet:
        # 'fact_check',
        # 'timeline',
        # 'legislative',
        # 'campaign_promises',
        # 'executive_actions',
        # 'media_coverage',
        # 'public_opinion',
        # 'controversies',
        # 'policy_comparison',
        # 'judicial_appointments',
        # 'foreign_policy',
        # 'economic_metrics',
        # 'charity',
    ]
    
    # Initialize each database
    for db_name in db_names:
        print(f"Initializing {db_name} database...")
        db = get_database(db_name)
        print(f"  {db_name} database initialized.")


def populate_biography_database():
    """Populate the Biography Database with initial data."""
    print("Populating Biography Database...")
    
    # Get the database
    db = get_database('biography')
    
    # Import data from Wikipedia
    trump_id = db.import_from_wikipedia("Donald Trump", "Donald Trump")
    if trump_id:
        print(f"  Imported Donald Trump biography (ID: {trump_id})")
        
        # Add additional biographical information for Trump
        db.add_biographical_event(
            politician_id=trump_id,
            event_date="1946-06-14",
            event_type="birth",
            event_description="Born in Queens, New York City",
            source_url="https://en.wikipedia.org/wiki/Donald_Trump"
        )
        
        # Add education information for Trump
        db.add_education(
            politician_id=trump_id,
            institution="Fordham University",
            start_year=1964,
            end_year=1966
        )
        
        db.add_education(
            politician_id=trump_id,
            institution="Wharton School of the University of Pennsylvania",
            degree="Bachelor of Science",
            field_of_study="Economics",
            start_year=1966,
            end_year=1968
        )
        
        # Add career information for Trump
        db.add_career(
            politician_id=trump_id,
            organization="Trump Organization",
            role="Chairman and President",
            start_year=1971,
            end_year=2017,
            description="Led the Trump Organization, a real estate development company"
        )
        
        db.add_career(
            politician_id=trump_id,
            organization="United States of America",
            role="45th President",
            start_year=2017,
            end_year=2021,
            description="Served as the 45th President of the United States"
        )
        
        # Add family information for Trump
        db.add_family_member(
            politician_id=trump_id,
            relationship="spouse",
            name="Melania Trump",
            birth_year=1970,
            description="First Lady of the United States (2017-2021)"
        )
    
    # Import data for Joe Biden
    biden_id = db.import_from_wikipedia("Joe Biden", "Joe Biden")
    if biden_id:
        print(f"  Imported Joe Biden biography (ID: {biden_id})")
        
        # Add additional biographical information for Biden
        db.add_biographical_event(
            politician_id=biden_id,
            event_date="1942-11-20",
            event_type="birth",
            event_description="Born in Scranton, Pennsylvania",
            source_url="https://en.wikipedia.org/wiki/Joe_Biden"
        )
        
        # Add education information for Biden
        db.add_education(
            politician_id=biden_id,
            institution="University of Delaware",
            degree="Bachelor of Arts",
            field_of_study="History and Political Science",
            start_year=1961,
            end_year=1965
        )
        
        db.add_education(
            politician_id=biden_id,
            institution="Syracuse University College of Law",
            degree="Juris Doctor",
            start_year=1965,
            end_year=1968
        )
        
        # Add career information for Biden
        db.add_career(
            politician_id=biden_id,
            organization="United States Senate",
            role="Senator from Delaware",
            start_year=1973,
            end_year=2009,
            description="Represented Delaware in the U.S. Senate"
        )
        
        db.add_career(
            politician_id=biden_id,
            organization="United States of America",
            role="47th Vice President",
            start_year=2009,
            end_year=2017,
            description="Served as Vice President under President Barack Obama"
        )
        
        db.add_career(
            politician_id=biden_id,
            organization="United States of America",
            role="46th President",
            start_year=2021,
            end_year=None,
            description="Serving as the 46th President of the United States"
        )
        
        # Add family information for Biden
        db.add_family_member(
            politician_id=biden_id,
            relationship="spouse",
            name="Jill Biden",
            birth_year=1951,
            description="First Lady of the United States (2021-present)"
        )


def populate_databases():
    """Populate all databases with initial data."""
    populate_biography_database()
    # Add functions to populate other databases here


if __name__ == "__main__":
    # Initialize all databases
    initialize_databases()
    
    # Populate databases with initial data
    populate_databases()
    
    print("All databases initialized and populated successfully.")