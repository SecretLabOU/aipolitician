"""Data collection and processing for the PoliticianAI project."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from sqlalchemy.orm import Session

from src.database.models import (
    init_db,
    Politician,
    PolicyPosition,
    VotingRecord,
    Statement
)
from src.utils.helpers import create_directory

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and processes political data from various sources."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        
        # Create directories
        create_directory(self.raw_dir)
        create_directory(self.processed_dir)

    def collect_huggingface_data(self):
        """Collect data from Hugging Face datasets."""
        datasets = [
            ("bananabot/TrumpSpeeches", "speeches"),
            ("yunfan-y/trump-tweets-cleaned", "tweets"),
            ("pookie3000/trump-interviews", "interviews")
        ]
        
        for dataset_name, data_type in datasets:
            logger.info(f"Collecting {data_type} from {dataset_name}...")
            
            try:
                # Load dataset
                dataset = load_dataset(dataset_name)
                
                # Save raw data
                output_file = self.raw_dir / f"{data_type}.json"
                dataset.save_to_disk(str(output_file))
                
                logger.info(f"Saved {data_type} to {output_file}")
                
            except Exception as e:
                logger.error(f"Error collecting {data_type}: {str(e)}")

    def collect_congress_data(self):
        """
        Collect data from congress.gov public dataset.
        This is a placeholder - in a real implementation, you would
        scrape or use an API to get this data.
        """
        # Sample voting data
        voting_data = [
            {
                "bill_name": "Healthcare Reform Act of 2023",
                "vote": "Yes",
                "date": "2023-06-15",
                "topic": "healthcare"
            },
            {
                "bill_name": "Climate Action Plan",
                "vote": "No",
                "date": "2023-07-20",
                "topic": "climate_change"
            },
            {
                "bill_name": "Infrastructure Investment Act",
                "vote": "Yes",
                "date": "2023-08-10",
                "topic": "infrastructure"
            }
        ]
        
        # Save raw data
        output_file = self.raw_dir / "voting_records.json"
        with open(output_file, 'w') as f:
            json.dump(voting_data, f, indent=2)
        
        logger.info(f"Saved voting records to {output_file}")

    def process_speeches(self):
        """Process collected speech data."""
        try:
            speeches_file = self.raw_dir / "speeches.json"
            if not speeches_file.exists():
                logger.warning("No speech data found")
                return
            
            # Load and process speeches
            dataset = load_dataset("json", data_files=str(speeches_file))
            
            for item in dataset['train']:
                statement = Statement(
                    content=item['text'],
                    date=datetime.fromisoformat(item['date']),
                    source_type='speech'
                )
                self.db.add(statement)
            
            self.db.commit()
            logger.info("Processed speeches successfully")
            
        except Exception as e:
            logger.error(f"Error processing speeches: {str(e)}")
            self.db.rollback()

    def process_voting_records(self):
        """Process voting records."""
        try:
            records_file = self.raw_dir / "voting_records.json"
            if not records_file.exists():
                logger.warning("No voting record data found")
                return
            
            with open(records_file, 'r') as f:
                records = json.load(f)
            
            for record in records:
                voting_record = VotingRecord(
                    bill_name=record['bill_name'],
                    vote=record['vote'],
                    date=datetime.fromisoformat(record['date']),
                    topic=record['topic']
                )
                self.db.add(voting_record)
            
            self.db.commit()
            logger.info("Processed voting records successfully")
            
        except Exception as e:
            logger.error(f"Error processing voting records: {str(e)}")
            self.db.rollback()

    def initialize_sample_data(self):
        """Initialize sample politician data."""
        try:
            # Sample politician data
            politicians = [
                {
                    "name": "John Smith",
                    "party": "Independent",
                    "current_position": "Senator",
                    "bio": "Experienced senator focused on economic policy"
                }
            ]
            
            # Sample policy positions
            positions = [
                {
                    "topic": "healthcare",
                    "position": "Supports universal healthcare with private options",
                    "date": "2024-01-01"
                },
                {
                    "topic": "climate_change",
                    "position": "Advocates for renewable energy transition",
                    "date": "2024-01-01"
                },
                {
                    "topic": "economy",
                    "position": "Promotes balanced approach to regulation",
                    "date": "2024-01-01"
                }
            ]
            
            # Add data to database
            for pol_data in politicians:
                politician = Politician(**pol_data)
                self.db.add(politician)
                self.db.flush()  # Get ID
                
                for pos in positions:
                    position = PolicyPosition(
                        politician_id=politician.id,
                        topic=pos['topic'],
                        position=pos['position'],
                        date_updated=datetime.fromisoformat(pos['date'])
                    )
                    self.db.add(position)
            
            self.db.commit()
            logger.info("Initialized sample data successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
            self.db.rollback()

def main():
    """Main data collection function."""
    # Initialize database
    engine, SessionLocal = init_db()
    db = SessionLocal()
    
    try:
        collector = DataCollector(db)
        
        # Collect data
        collector.collect_huggingface_data()
        collector.collect_congress_data()
        
        # Process data
        collector.process_speeches()
        collector.process_voting_records()
        
        # Initialize sample data
        collector.initialize_sample_data()
        
        logger.info("Data collection and processing complete!")
        
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}")
        raise
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
