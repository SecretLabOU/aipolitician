#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#!/usr/bin/env python3
"""
Insert sample data into the Milvus database.
This script adds a test political figure to verify database functionality.
"""

import sys
import json
import uuid
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from scripts.schema import connect_to_milvus, create_political_figures_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_sample_data():
    """Insert a sample political figure record"""
    # Connect to Milvus
    if not connect_to_milvus():
        logger.error("Failed to connect to Milvus")
        return False
    
    # Get the collection
    collection = create_political_figures_collection()
    
    # Load the embedding model that produces 768-dimensional vectors
    model = SentenceTransformer('all-mpnet-base-v2')  # This model produces 768-dimensional vectors
    
    # Sample data for Joe Biden
    sample_data = {
        "id": str(uuid.uuid4()),
        "name": "Joe Biden",
        "date_of_birth": "1942-11-20",
        "nationality": "American",
        "political_affiliation": "Democratic Party",
        "biography": "Joseph Robinette Biden Jr. is an American politician who is the 46th president of the United States. A member of the Democratic Party, he previously served as the 47th vice president from 2009 to 2017 under President Barack Obama and represented Delaware in the United States Senate from 1973 to 2009.",
        "positions": json.dumps([
            {"title": "President of the United States", "start_year": 2021, "end_year": None},
            {"title": "Vice President of the United States", "start_year": 2009, "end_year": 2017},
            {"title": "U.S. Senator from Delaware", "start_year": 1973, "end_year": 2009}
        ]),
        "policies": json.dumps({
            "healthcare": "Supporter of the Affordable Care Act",
            "climate": "Committed to addressing climate change, rejoined Paris Agreement",
            "economy": "Focus on middle-class growth and infrastructure investment"
        }),
        "legislative_actions": json.dumps([
            {"title": "American Rescue Plan", "year": 2021, "description": "COVID-19 economic stimulus package"},
            {"title": "Infrastructure Investment and Jobs Act", "year": 2021, "description": "Bipartisan infrastructure bill"}
        ]),
        "public_communications": json.dumps({
            "speeches": ["Inaugural Address", "State of the Union"],
            "interviews": ["60 Minutes", "CNN Town Hall"]
        }),
        "timeline": json.dumps([
            {"year": 1942, "event": "Born in Scranton, Pennsylvania"},
            {"year": 1972, "event": "Elected to the U.S. Senate"},
            {"year": 2009, "event": "Became Vice President"},
            {"year": 2021, "event": "Inaugurated as 46th President"}
        ]),
        "campaigns": json.dumps([
            {"year": 1988, "position": "President", "outcome": "Withdrew during primaries"},
            {"year": 2008, "position": "President", "outcome": "Withdrew during primaries"},
            {"year": 2020, "position": "President", "outcome": "Won general election"}
        ]),
        "media": json.dumps({
            "books": ["Promise Me, Dad", "Promises to Keep"],
            "documentaries": ["Joe Biden: American Dreamer"]
        }),
        "philanthropy": json.dumps({
            "foundations": ["Biden Foundation", "Biden Cancer Initiative"],
            "causes": ["Cancer research", "Violence against women"]
        }),
        "personal_details": json.dumps({
            "religion": "Roman Catholic",
            "education": "University of Delaware, Syracuse University Law School",
            "family": "Married to Jill Biden, father to Hunter, Ashley, and the late Beau and Naomi"
        })
    }
    
    # Generate embedding from biography
    sample_data["embedding"] = model.encode(sample_data["biography"]).tolist()
    
    # Insert the data
    try:
        collection.insert([sample_data])
        logger.info(f"Sample record for {sample_data['name']} inserted successfully")
        logger.info(f"Record ID: {sample_data['id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to insert sample record: {str(e)}")
        return False

if __name__ == "__main__":
    print("Inserting sample data into Milvus database...")
    if insert_sample_data():
        print("Sample data inserted successfully!")
    else:
        print("Failed to insert sample data.")
        sys.exit(1)
