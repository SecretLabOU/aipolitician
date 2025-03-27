# -*- coding: utf-8 -*-
import os
import json
import datetime
import uuid
import spacy
from spacy.util import get_cuda_devices
from spacy.tokens import Doc


class SpacyNERPipeline:
    """
    Pipeline for processing text with SpaCy NER and extracting structured information
    """
    
    def __init__(self, model_name, politician_data_dir):
        self.model_name = model_name
        self.politician_data_dir = politician_data_dir
        self.nlp = None
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            model_name=crawler.settings.get('SPACY_MODEL', 'en_core_web_lg'),
            politician_data_dir=crawler.settings.get('POLITICIAN_DATA_DIR', 'data/politicians')
        )
    
    def open_spider(self, spider):
        """Initialize SpaCy model and optimize for GPU if available"""
        print(f"Loading SpaCy model: {self.model_name}")
        
        # Check for GPU availability and optimize settings
        has_gpu = len(get_cuda_devices()) > 0
        if has_gpu:
            spacy.prefer_gpu()
            print("✅ GPU is available and will be used by SpaCy")
            
            # Load model with GPU optimization
            self.nlp = spacy.load(self.model_name)
            
            # Set batch size larger for GPU
            Doc.set_extension("politician_name", default=None, force=True)
            self.nlp.batch_size = 128
        else:
            print("⚠️ No GPU available, using CPU for NLP processing")
            self.nlp = spacy.load(self.model_name)
            Doc.set_extension("politician_name", default=None, force=True)
            self.nlp.batch_size = 32
        
        # Create output directory
        os.makedirs(self.politician_data_dir, exist_ok=True)
        
        print(f"✅ Successfully initialized SpaCy pipeline with {self.model_name}")
    
    def process_item(self, item, spider):
        """Process text with SpaCy NER and extract relevant entities"""
        if not item.get('raw_content'):
            return item
        
        # Process text with SpaCy
        doc = self.nlp(item['raw_content'])
        
        # Set politician name for reference
        doc._.politician_name = item['name']
        
        # Extract named entities by type
        person_entities = []
        org_entities = []
        date_entities = []
        gpe_entities = []
        event_entities = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_entities.append(ent.text)
            elif ent.label_ == "ORG":
                org_entities.append(ent.text)
            elif ent.label_ == "DATE":
                date_entities.append(ent.text)
            elif ent.label_ == "GPE":
                gpe_entities.append(ent.text)
            elif ent.label_ == "EVENT":
                event_entities.append(ent.text)
        
        # Update item with extracted entities
        item['person_entities'] = list(set(person_entities))
        item['org_entities'] = list(set(org_entities))
        item['date_entities'] = list(set(date_entities))
        item['gpe_entities'] = list(set(gpe_entities))
        item['event_entities'] = list(set(event_entities))
        
        # Extract basic information based on entities
        # Find potential birth dates
        if not item.get('date_of_birth') and date_entities:
            # Look for date entities near birth-related terms
            birth_dates = []
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if any(term in sent_text for term in ['born', 'birth', 'birthday']):
                    for ent in sent.ents:
                        if ent.label_ == "DATE":
                            birth_dates.append(ent.text)
            
            if birth_dates:
                item['date_of_birth'] = birth_dates[0]
        
        # Find political affiliation
        if not item.get('political_affiliation') and org_entities:
            political_orgs = ["Republican", "Democrat", "Democratic", "GOP", "Conservative", "Liberal", "Labour", "Tory"]
            
            for org in org_entities:
                if any(party in org for party in political_orgs):
                    item['political_affiliation'] = org
                    break
        
        # Add timestamp and ID if not present
        if not item.get('timestamp'):
            item['timestamp'] = datetime.datetime.now().isoformat()
        
        if not item.get('id'):
            item['id'] = str(uuid.uuid4())
            
        return item


class JsonWriterPipeline:
    """
    Pipeline for saving scraped items to JSON files
    """
    
    def __init__(self, politician_data_dir):
        self.politician_data_dir = politician_data_dir
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            politician_data_dir=crawler.settings.get('POLITICIAN_DATA_DIR', 'data/politicians')
        )
    
    def open_spider(self, spider):
        self.files = {}
        os.makedirs(self.politician_data_dir, exist_ok=True)
    
    def close_spider(self, spider):
        for file in self.files.values():
            file.close()
    
    def process_item(self, item, spider):
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in item['name'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        filepath = os.path.join(self.politician_data_dir, filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(item), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved politician data to: {filepath}")
        return item 