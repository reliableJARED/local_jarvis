# search.py
from ddgs import DDGS
from bs4 import BeautifulSoup
from bs4.element import Comment

import urllib.parse
import random
import asyncio
from playwright.async_api import async_playwright

from bart_lg import TextSummarizer

class WebSearch:
    def __init__(self):
        self.search = DDGS()
        self.summarizer = TextSummarizer()

    def tag_visible(self,element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        if element.name == 'br':
            return True  # Include <br> tags for line breaks
        return True

    def text_from_html(self,html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(self.tag_visible, texts)
        website_text = " ".join(t.strip() for t in visible_texts)
        return website_text
    
    def summarize_text(self,text):
        summary = self.summarizer.summarize_text(text)
        return summary

    async def _fetch_with_playwright(self, url, wait_time=2):
        """
        Fetch content from a URL using Playwright with cookie handling and realistic browser behavior
        
        Args:
            url: URL to fetch
            wait_time: Seconds to wait after page load for dynamic content
        
        Returns:
            HTML content as string, or None if failed
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ]
            
            context = await browser.new_context(
                user_agent=random.choice(user_agents),
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
                accept_downloads=False,
            )
            
            page = await context.new_page()
            
            try:
                # Navigate with realistic behavior - wait for network idle
                print(f"Fetching with Playwright: {url}")
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait a bit for any dynamic content to load
                await asyncio.sleep(wait_time)
                
                # Get the page content
                content = await page.content()
                print(f"✓ Successfully fetched content from {url}")
                
                return content
                
            except Exception as e:
                print(f"✗ Error fetching {url}: {type(e).__name__}: {e}")
                return None
            
            finally:
                await browser.close()

    async def _google_search_playwright(self, query, max_results=10):
        """
        Perform a Google search using Playwright with JavaScript rendering
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
        
        Returns:
            List of dictionaries containing search results
        """
        print(f"Searching Google for: {query}")
        print("-" * 50)
        
        async with async_playwright() as p:
            # Launch browser with anti-detection features
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            # Create context with realistic settings using random user agent
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ]
            
            context = await browser.new_context(
                user_agent=random.choice(user_agents),
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York',
            )
            
            # Create new page
            page = await context.new_page()
            
            try:
                # Navigate to Google search
                search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num={max_results}"
                print(f"Navigating to: {search_url}")
                
                await page.goto(search_url, wait_until='networkidle')
                print("✓ Page loaded successfully")
                
                # Wait for search results to appear
                await page.wait_for_selector('div#search', timeout=10000)
                print("✓ Search results container found")
                
                # Extract search results
                results = []
                
                # Try multiple selector strategies
                result_selectors = [
                    'div.g',  # Standard result container
                    'div.tF2Cxc',  # 2024-2025 structure
                ]
                
                containers = []
                for selector in result_selectors:
                    containers = await page.query_selector_all(selector)
                    if containers:
                        print(f"✓ Found {len(containers)} results with selector: {selector}")
                        break
                
                if not containers:
                    print("✗ No result containers found")
                    return []
                
                for container in containers:
                    try:
                        # Extract title
                        title_elem = await container.query_selector('h3')
                        if not title_elem:
                            continue
                        title = await title_elem.inner_text()
                        
                        # Extract URL
                        link_elem = await container.query_selector('a')
                        if not link_elem:
                            continue
                        url = await link_elem.get_attribute('href')
                        
                        # Skip invalid links
                        if not url or url.startswith('#') or '/search?' in url:
                            continue
                        
                        # Extract snippet
                        snippet = ''
                        snippet_selectors = ['.VwiC3b', '.lyLwlc', '.s3v9rd', 'div[data-sncf="1"]']
                        for snip_sel in snippet_selectors:
                            snippet_elem = await container.query_selector(snip_sel)
                            if snippet_elem:
                                snippet = await snippet_elem.inner_text()
                                break
                        
                        results.append({
                            'title': title.strip(),
                            'href': url,
                            'body': snippet.strip()
                        })
                        
                    except Exception as e:
                        # Skip problematic results
                        continue
                
                print(f"✓ Successfully extracted {len(results)} results")
                return results
                
            except Exception as e:
                print(f"✗ Error: {type(e).__name__}: {e}")
                return []
            
            finally:
                await browser.close()

    def askInternet_google(self, query, max_results=5):
        """
        Perform a Google search using Playwright and process results with summarization
        
        Args:
            query: Search query string
            max_results: Maximum number of results to fetch and process
        
        Returns:
            List of summaries from search results
        """
        # Run the async Google search
        results = asyncio.run(self._google_search_playwright(query, max_results))
        
        print(results)
        print("="*50)
        
        # Show all the results
        for res in results:
            print(res['href'])
        print("="*50)
        
        all_summaries = []
        
        # Process each result
        for result in results:
            url = result['href']
            snippet = result['body']
            
            # Check if URL points to a PDF
            if url.lower().endswith('.pdf'):
                print(f"Processing PDF from: {url}\n")
                try:
                    import requests
                    import PyPDF2
                    from io import BytesIO
                    
                    # Download the PDF
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    # Extract text from PDF
                    pdf_file = BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    # Summarize the PDF content
                    summary = self.summarizer.summarize_text(pdf_text, query_match=query)
                    print(f"{snippet}: {summary}")
                    all_summaries.append(f"{snippet}: {summary}")
                    
                except Exception as e:
                    print(f"Error processing PDF {url}: {e}")
                    all_summaries.append(snippet)  # Fall back to snippet
                
                continue
            
            # Use Playwright for HTML content fetching
            print(f"Fetching content from: {url}\n\n")
            html_content = asyncio.run(self._fetch_with_playwright(url))
            
            if html_content is None:
                all_summaries.append(snippet)
                continue
            
            website_text = self.text_from_html(html_content)
            summary = self.summarizer.summarize_text(website_text, query_match=query)
            print(snippet + ":" + summary)
            all_summaries.append(snippet + ":" + summary)
        
        return all_summaries

    
    def askInternet(self, query, max_results=5):
        # Perform the search https://pypi.org/project/ddgs/
        results = self.search.text(query, backend="bing,yahoo,duckduckgo,brave, yandex", max_results=max_results)
        print(results)
        print("="*50)
        # Show all the results
        for res in results:
            print(res['href'])
        print("="*50)
        all_summaries = []
        
        # Process each result
        for result in results:
            url = result['href']
            snippet = result['body']
            
            # Check if URL points to a PDF
            if url.lower().endswith('.pdf'):
                print(f"Processing PDF from: {url}\n")
                try:
                    import requests
                    import PyPDF2
                    from io import BytesIO
                    
                    # Download the PDF
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    # Extract text from PDF
                    pdf_file = BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        pdf_text += page.extract_text()
                    
                    # Summarize the PDF content
                    summary = self.summarizer.summarize_text(pdf_text, query_match=query)
                    print(f"{snippet}: {summary}")
                    all_summaries.append(f"{snippet}: {summary}")
                    
                except Exception as e:
                    print(f"Error processing PDF {url}: {e}")
                    all_summaries.append(snippet)  # Fall back to snippet
                
                continue
            
            # Use Playwright for HTML content fetching
            print(f"Fetching content from: {url}\n\n")
            html_content = asyncio.run(self._fetch_with_playwright(url))
            
            if html_content is None:
                all_summaries.append(snippet)
                continue
            
            website_text = self.text_from_html(html_content)
            summary = self.summarizer.summarize_text(website_text, query_match=query)
            print(snippet + ":" + summary)
            all_summaries.append(snippet + ":" + summary)
        
        return all_summaries

if __name__ == "__main__":
    
    ws = WebSearch()
    
    #summary = ws.askInternet("did the Patriots win today?")
    summary = ws.askInternet_google("did the Patriots win today?")
    print("\n\nALL SUMMARIES:\n\n")
    print(summary,"\n\n",len(summary))

    x = ws.summarize_text("".join(summary))
    print("\n\nSUMMARY OF ALL SUMMARIES:\n\n")
    print(x)