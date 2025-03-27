# -*- coding: utf-8 -*-
import scrapy
import re
import logging
from urllib.parse import urljoin
from src.data.scraper.politician_crawler.items import PoliticianItem


class PoliticianSpider(scrapy.Spider):
    """
    Spider for crawling politician information from various sources
    """
    name = "politician"
    allowed_domains = [
        'en.wikipedia.org',
        'www.britannica.com',
        'www.reuters.com',
        'apnews.com',
        'www.bbc.com',
        'www.npr.org',
        'www.cnn.com',
        'www.foxnews.com',
        'congress.gov',
        'www.govtrack.us',
        'www.c-span.org',
        'millercenter.org',
        'votesmart.org',
        'www.politifact.com',
        'www.factcheck.org'
    ]
    
    def __init__(self, politician_name=None, *args, **kwargs):
        super(PoliticianSpider, self).__init__(*args, **kwargs)
        
        # Ensure politician name is provided
        if not politician_name:
            raise ValueError("Politician name must be provided using the -a politician_name='Name' parameter")
            
        self.politician_name = politician_name
        self.logger.info(f"Starting spider for politician: {politician_name}")
        
        # Format name for different URL patterns
        self.formatted_name = politician_name.replace(' ', '_')
        self.formatted_name_dash = politician_name.replace(' ', '-')
        self.formatted_name_plus = politician_name.replace(' ', '+')
        
        # Generate start URLs
        self.start_urls = self.generate_start_urls()
    
    def generate_start_urls(self):
        """Generate URLs for various sources based on politician name"""
        name = self.politician_name
        formatted_name = self.formatted_name
        formatted_name_dash = self.formatted_name_dash
        formatted_name_plus = self.formatted_name_plus
        
        return [
            # Wikipedia and encyclopedias
            f"https://en.wikipedia.org/wiki/{formatted_name}",
            f"https://www.britannica.com/biography/{formatted_name_dash}",
            
            # News sources
            f"https://www.reuters.com/search/news?blob={name}",
            f"https://apnews.com/search?q={formatted_name_plus}",
            f"https://www.bbc.com/news/topics/{formatted_name_dash}",
            f"https://www.npr.org/search?query={formatted_name_plus}",
            f"https://www.cnn.com/search?q={formatted_name_plus}",
            f"https://www.foxnews.com/search-results/{formatted_name_plus}",
            
            # Government sources
            f"https://www.congress.gov/search?q=%7B%22source%22%3A%22members%22%2C%22search%22%3A%22{formatted_name_plus}%22%7D",
            f"https://www.govtrack.us/congress/members/find?q={formatted_name_plus}",
            
            # Speech archives
            f"https://www.c-span.org/search/?query={formatted_name_plus}",
            f"https://millercenter.org/the-presidency/presidential-speeches?field_president_target_id=All&keys={formatted_name_plus}",
            
            # Voting records
            f"https://www.govtrack.us/congress/votes/presidential-candidates",
            f"https://votesmart.org/candidate/key-votes/{formatted_name_dash}",
            
            # Fact-checking sites
            f"https://www.politifact.com/personalities/{formatted_name_dash}/",
            f"https://www.factcheck.org/?s={formatted_name_plus}",
        ]
    
    def parse(self, response):
        """Parse the response based on the domain"""
        self.logger.info(f"Parsing URL: {response.url}")
        
        # Determine which domain-specific parser to use
        if "wikipedia.org" in response.url:
            yield from self.parse_wikipedia(response)
        elif "britannica.com" in response.url:
            yield from self.parse_britannica(response)
        elif "reuters.com" in response.url:
            yield from self.parse_news_search(response)
        elif "apnews.com" in response.url:
            yield from self.parse_news_search(response)
        elif "congress.gov" in response.url:
            yield from self.parse_congress(response)
        elif "govtrack.us" in response.url:
            yield from self.parse_govtrack(response)
        elif "c-span.org" in response.url:
            yield from self.parse_cspan(response)
        elif "votesmart.org" in response.url:
            yield from self.parse_votesmart(response)
        elif "politifact.com" in response.url:
            yield from self.parse_politifact(response)
        else:
            # Generic parser for other sites
            yield from self.parse_generic(response)
        
        # Follow next page links if available and within same domain
        next_page = response.css('a.next::attr(href), a.pagination-next::attr(href), '
                               'a[rel="next"]::attr(href), a:contains("Next")::attr(href)').get()
        
        if next_page:
            next_url = urljoin(response.url, next_page)
            if any(domain in next_url for domain in self.allowed_domains):
                self.logger.info(f"Following next page: {next_url}")
                yield scrapy.Request(next_url, callback=self.parse)
    
    def parse_wikipedia(self, response):
        """Parse Wikipedia articles"""
        # Check if it's a disambiguation page
        if "may refer to" in response.css("div#mw-content-text").get():
            # Follow the first relevant link
            links = response.css("div#mw-content-text ul li a")
            for link in links:
                if self.politician_name.lower() in link.css("::text").get().lower():
                    next_url = urljoin(response.url, link.css("::attr(href)").get())
                    yield scrapy.Request(next_url, callback=self.parse)
                    break
            return
        
        # Extract content
        content_div = response.css('div#mw-content-text')
        
        if not content_div:
            self.logger.warning(f"No content found on Wikipedia page: {response.url}")
            return
        
        # Extract main paragraphs, excluding tables and references
        paragraphs = content_div.css('p').getall()[:25]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        
        # Try to extract basic info from infobox
        infobox = response.css('table.infobox')
        if infobox:
            # Birth date
            born_row = infobox.css('th:contains("Born") + td').get()
            if born_row:
                birth_date = re.search(r'(\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},\s+\d{4})', self.clean_html(born_row))
                if birth_date:
                    item['date_of_birth'] = birth_date.group(1)
            
            # Political party
            party_row = infobox.css('th:contains("Political party") + td').get()
            if party_row:
                item['political_affiliation'] = self.clean_html(party_row)
        
        yield item
    
    def parse_britannica(self, response):
        """Parse Britannica biography pages"""
        content_div = response.css('div.topic-content')
        
        if not content_div:
            self.logger.warning(f"No content found on Britannica page: {response.url}")
            return
        
        # Extract main paragraphs
        paragraphs = content_div.css('p').getall()[:15]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        yield item
    
    def parse_news_search(self, response):
        """Parse news search results pages and follow article links"""
        # Extract article links
        article_links = response.css('article a::attr(href), .story-content a::attr(href), .searchResult a::attr(href)').getall()
        
        # Follow each article link with politician name
        for link in article_links:
            full_url = urljoin(response.url, link)
            # Only follow if it's not a search page and is within allowed domains
            if not re.search(r'search', full_url) and any(domain in full_url for domain in self.allowed_domains):
                yield scrapy.Request(full_url, callback=self.parse_news_article)
    
    def parse_news_article(self, response):
        """Parse individual news articles"""
        # Extract article content
        article = response.css('article, .article-body, .story-content')
        
        if not article:
            self.logger.warning(f"No article content found on: {response.url}")
            return
        
        paragraphs = article.css('p').getall()
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        yield item
    
    def parse_congress(self, response):
        """Parse Congress.gov pages"""
        main_content = response.css('div#main, div.main-wrapper')
        
        if not main_content:
            self.logger.warning(f"No main content found on Congress page: {response.url}")
            return
        
        # Extract content
        paragraphs = main_content.css('p, li, h2, h3').getall()[:20]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        
        # Extract bills if available
        bills = []
        for bill in response.css('td.actions-result-content a::text').getall():
            bills.append(bill)
        
        if bills:
            item['sponsored_bills'] = bills
        
        yield item
    
    def parse_govtrack(self, response):
        """Parse GovTrack pages"""
        main_content = response.css('div.tab-pane, div#tab-details, div.row-fluid')
        
        if not main_content:
            self.logger.warning(f"No main content found on GovTrack page: {response.url}")
            return
        
        # Extract content
        paragraphs = main_content.css('p, li, h4, div.col-sm-10').getall()[:20]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        
        # Extract voting records if available
        votes = []
        for vote in response.css('table.vote-details tr'):
            vote_text = self.clean_html(vote.get())
            if vote_text.strip():
                votes.append(vote_text)
        
        if votes:
            item['voting_record'] = votes
        
        yield item
    
    def parse_cspan(self, response):
        """Parse C-SPAN pages"""
        main_content = response.css('div.video-content, div.player-holder')
        
        if not main_content:
            self.logger.warning(f"No main content found on C-SPAN page: {response.url}")
            return
        
        # Extract content
        paragraphs = main_content.css('p, li').getall()[:20]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        
        # Extract speech transcripts if available
        transcript = response.css('div.transcript-line').getall()
        if transcript:
            speeches = []
            for line in transcript:
                clean_line = self.clean_html(line)
                if clean_line.strip():
                    speeches.append(clean_line)
            
            if speeches:
                item['speeches'] = speeches
        
        yield item
    
    def parse_votesmart(self, response):
        """Parse VoteSmart pages"""
        main_content = response.css('div.breakdown, div.candidate')
        
        if not main_content:
            self.logger.warning(f"No main content found on VoteSmart page: {response.url}")
            return
        
        # Extract content
        paragraphs = main_content.css('p, li, div, span').getall()[:30]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        yield item
    
    def parse_politifact(self, response):
        """Parse PolitiFact pages"""
        main_content = response.css('div.m-statements')
        
        if not main_content:
            self.logger.warning(f"No main content found on PolitiFact page: {response.url}")
            return
        
        # Extract content
        items = main_content.css('div, p').getall()[:20]
        raw_content = "\n".join([self.clean_html(i) for i in items])
        
        # Create item
        item = PoliticianItem()
        item['name'] = self.politician_name
        item['source_url'] = response.url
        item['raw_content'] = raw_content
        
        # Extract statements if available
        statements = []
        for statement in response.css('div.m-statement__quote'):
            statement_text = self.clean_html(statement.get())
            if statement_text.strip():
                statements.append(statement_text)
        
        if statements:
            item['statements'] = statements
        
        yield item
    
    def parse_generic(self, response):
        """Generic parser for any other site"""
        # Extract main content - look for common content containers
        main_content = response.css('article, .article, .content, main, #main, .main-content')
        
        if not main_content:
            # Fallback to body if no specific content container
            main_content = response.css('body')
        
        # Extract paragraphs and headers
        paragraphs = main_content.css('p, h1, h2, h3, h4, li').getall()[:30]
        raw_content = "\n".join([self.clean_html(p) for p in paragraphs])
        
        # Only create an item if we have sufficient content
        if len(raw_content) > 100:
            item = PoliticianItem()
            item['name'] = self.politician_name
            item['source_url'] = response.url
            item['raw_content'] = raw_content
            yield item
    
    def clean_html(self, html_text):
        """Remove HTML tags and clean up the text"""
        if not html_text:
            return ""
            
        # Use regex to remove HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', html_text)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Remove citations [1], [2], etc.
        clean_text = re.sub(r'\[\d+\]', '', clean_text)
        
        return clean_text 