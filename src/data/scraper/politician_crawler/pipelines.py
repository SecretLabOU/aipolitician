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
            politician_data_dir=crawler.settings.get('POLITICIAN_DATA_DIR', 'src/data/scraper/logs')
        )
    
    def open_spider(self, spider):
        """Initialize SpaCy model and optimize for GPU if available"""
        print(f"Loading SpaCy model: {self.model_name}")
        
        try:
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
        except OSError as e:
            print(f"❌ ERROR: Could not load SpaCy model '{self.model_name}': {e}")
            print("⚠️ Run: python -m spacy download en_core_web_lg")
            # Set to None and handle in process_item
            self.nlp = None
        except Exception as e:
            print(f"❌ ERROR initializing SpaCy: {e}")
            self.nlp = None
    
    def process_item(self, item, spider):
        """Process text with SpaCy NER and extract relevant entities"""
        if not item.get('raw_content'):
            return item
        
        # If NLP is not initialized, just pass the item through
        if self.nlp is None:
            print("⚠️ SpaCy NLP not initialized, skipping NER processing")
            # Still add timestamp and ID
            if not item.get('timestamp'):
                item['timestamp'] = datetime.datetime.now().isoformat()
            if not item.get('id'):
                item['id'] = str(uuid.uuid4())
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
        self.saved_files = []
    
    @classmethod
    def from_crawler(cls, crawler):
        output_dir = crawler.settings.get('POLITICIAN_DATA_DIR', 'src/data/scraper/logs')
        # Only convert to absolute path if not already absolute
        if not os.path.isabs(output_dir):
            # Get project root directory
            project_root = os.getcwd()
            output_dir = os.path.join(project_root, output_dir)
        
        # Try to immediately create the directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"💾 Verified output directory exists: {output_dir}")
            # Check if we can write to it
            test_file = os.path.join(output_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"✅ Successfully verified write permission for: {output_dir}")
            else:
                print(f"⚠️ Warning: Could not verify file creation in {output_dir}")
        except Exception as e:
            print(f"❌ ERROR: Could not create or write to output directory: {output_dir}")
            print(f"❌ Error details: {str(e)}")
        
        return cls(politician_data_dir=output_dir)
    
    def open_spider(self, spider):
        self.files = {}
        # Log the absolute path of the output directory
        print(f"🗂️ Output directory absolute path: {os.path.abspath(self.politician_data_dir)}")
        try:
            os.makedirs(self.politician_data_dir, exist_ok=True)
            print(f"📁 Directory permissions: {oct(os.stat(self.politician_data_dir).st_mode)[-3:]}")
        except Exception as e:
            print(f"❌ Failed to create directory: {str(e)}")
    
    def close_spider(self, spider):
        for file in self.files.values():
            file.close()
        
        # Final check of saved files
        print(f"🔍 Checking saved files in {self.politician_data_dir}:")
        try:
            if os.path.exists(self.politician_data_dir):
                files = os.listdir(self.politician_data_dir)
                json_files = [f for f in files if f.endswith('.json')]
                if json_files:
                    print(f"✅ Found {len(json_files)} JSON files in output directory")
                    for jfile in json_files:
                        full_path = os.path.join(self.politician_data_dir, jfile)
                        file_size = os.path.getsize(full_path)
                        print(f"   - {jfile}: {file_size} bytes")
                else:
                    print(f"⚠️ WARNING: No JSON files found in {self.politician_data_dir}")
                    print(f"   All files in directory: {files}")
            else:
                print(f"❌ ERROR: Output directory does not exist: {self.politician_data_dir}")
        except Exception as e:
            print(f"❌ ERROR checking output directory: {str(e)}")
        
        if self.saved_files:
            print(f"📊 Summary of saved files:")
            for saved_file in self.saved_files:
                if os.path.exists(saved_file):
                    print(f"   ✅ {saved_file} exists, size: {os.path.getsize(saved_file)} bytes")
                else:
                    print(f"   ❌ {saved_file} does not exist!")
        else:
            print("⚠️ No files were processed by the pipeline!")
    
    def process_item(self, item, spider):
        if not item:
            print("❌ ERROR: Empty item received by JsonWriterPipeline")
            return item
            
        if 'name' not in item:
            print(f"❌ ERROR: Item has no 'name' field: {item}")
            return item
        
        print(f"💾 Processing item for {item['name']} from {item.get('source_url', 'unknown source')}")
        print(f"📋 Item fields: {list(item.keys())}")
            
        # Create a safe filename
        safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in item['name'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.json"
        
        # Don't create additional subdirectories, just use the file directly in politician_data_dir
        filepath = os.path.join(self.politician_data_dir, filename)
        
        # Log more detailed file path information
        print(f"📝 Attempting to save data to file: {os.path.abspath(filepath)}")
        print(f"📊 Item contains {len(item.keys())} fields")
        
        # Dump the full item data to help with debugging
        print(f"🔍 Item data preview: {str(dict(item))[:200]}...")
        
        # Ensure the directory exists
        try:
            os.makedirs(self.politician_data_dir, exist_ok=True)
            print(f"✅ Confirmed directory exists: {self.politician_data_dir}")
            print(f"📂 Directory absolute path: {os.path.abspath(self.politician_data_dir)}")
            print(f"📂 Directory contents: {os.listdir(self.politician_data_dir)}")
        except Exception as e:
            print(f"❌ ERROR creating directory: {self.politician_data_dir}")
            print(f"❌ Error details: {str(e)}")
            
        try:
            # Convert item to a regular dictionary
            item_dict = dict(item)
            
            # Check if the item dictionary is valid
            if not item_dict:
                print("❌ ERROR: Empty dictionary after conversion")
                return item
                
            # Write json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item_dict, f, indent=2, ensure_ascii=False)
            
            # Remember this file
            self.saved_files.append(filepath)
            
            # Check if file exists immediately after writing
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"✅ Successfully saved: {filepath} (Size: {file_size} bytes)")
            else:
                print(f"❌ ERROR: File does not exist after saving: {filepath}")
                
            # Try reading the file back to verify it worked
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    _ = json.load(f)
                print(f"✅ Verified file can be read back: {filepath}")
            except Exception as e:
                print(f"❌ ERROR reading back file: {str(e)}")
                
        except PermissionError as e:
            print(f"❌ Permission ERROR saving to file: {filepath}")
            print(f"❌ Error details: {str(e)}")
        except OSError as e:
            print(f"❌ OS ERROR saving to file: {filepath}")
            print(f"❌ Error details: {str(e)}")
        except Exception as e:
            print(f"❌ ERROR saving data to file: {filepath}")
            print(f"❌ Error details: {str(e)}")
        
        return item 