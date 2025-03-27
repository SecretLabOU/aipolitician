# -*- coding: utf-8 -*-
import scrapy

class PoliticianItem(scrapy.Item):
    """
    Item to store politician data extracted from various sources
    """
    # Basic identification
    id = scrapy.Field()
    name = scrapy.Field()
    source_url = scrapy.Field()
    
    # Basic info
    full_name = scrapy.Field()
    date_of_birth = scrapy.Field()
    place_of_birth = scrapy.Field()
    nationality = scrapy.Field()
    political_affiliation = scrapy.Field()
    education = scrapy.Field()
    family = scrapy.Field()
    
    # Career information
    positions = scrapy.Field()
    pre_political_career = scrapy.Field()
    committees = scrapy.Field()
    
    # Policy positions
    economy = scrapy.Field()
    foreign_policy = scrapy.Field()
    healthcare = scrapy.Field()
    immigration = scrapy.Field()
    environment = scrapy.Field()
    social_issues = scrapy.Field()
    other_policies = scrapy.Field()
    
    # Legislative record
    sponsored_bills = scrapy.Field()
    voting_record = scrapy.Field()
    achievements = scrapy.Field()
    
    # Communications
    speeches = scrapy.Field()
    statements = scrapy.Field()
    publications = scrapy.Field()
    
    # Campaigns
    elections = scrapy.Field()
    platforms = scrapy.Field()
    fundraising = scrapy.Field()
    
    # Controversies
    legal_issues = scrapy.Field()
    scandals = scrapy.Field()
    
    # Timeline
    timeline = scrapy.Field()
    
    # Raw content
    raw_content = scrapy.Field()
    
    # SpaCy NER extracted entities
    person_entities = scrapy.Field()
    org_entities = scrapy.Field()
    date_entities = scrapy.Field()
    gpe_entities = scrapy.Field()  # Geopolitical entities
    event_entities = scrapy.Field()
    
    # Metadata
    timestamp = scrapy.Field()
    spider = scrapy.Field()
    language = scrapy.Field() 