
from ddgs import DDGS
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import urllib.error

import random

import threading


from llm_tool_libs.bart_lg import TextSummarizer

class WebSearch:
    def __init__(self):
        self.search = DDGS()
        self.summarizer = TextSummarizer(keep_loaded_in_memory=True)

    def open_url(self, url):
        user_agents = [
            # Chrome on Windows:
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Chrome on macOS:
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Chrome on Linux:
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        ]
        headers = {
            'User-Agent': random.choice(user_agents)
        }
        try:
            page = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(page)
            html = response.read()
            return html
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code} while fetching {url}. Reason: {e.reason}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while fetching {url}: {e}")
            return None

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        if element.name == 'br':
            return True  # Include <br> tags for line breaks
        return True

    def text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(self.tag_visible, texts)
        website_text = " ".join(t.strip() for t in visible_texts)
        return website_text
    
    def summarize_text(self, text):
        summary = self.summarizer.summarize_text(text)
        return summary

    def _process_single_result(self, result, query, all_summaries, index):
        """
        Process a single search result (PDF or HTML) and add summary to shared list.
        
        Args:
            result: Dictionary containing 'href' and 'body' keys
            query: Original search query for matching
            all_summaries: Shared list to append results (thread-safe with proper indexing)
            index: Index position for this result in the summaries list
        """
        url = result['href']
        snippet = result['body']
        summary = None
        
        try:
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
                    summary = f"{snippet}: {summary}"
                    
                except Exception as e:
                    print(f"Error processing PDF {url}: {e}")
                    summary = snippet  # Fall back to snippet
            else:
                # Regular HTML processing
                print(f"Fetching content from: {url}\n")
                html_content = self.open_url(url)
                
                if html_content is None:
                    summary = snippet
                else:
                    website_text = self.text_from_html(html_content)
                    text_summary = self.summarizer.summarize_text(website_text, query_match=query)
                    print(f"{snippet}: {text_summary}")
                    summary = f"{snippet}: {text_summary}"
        
        except Exception as e:
            print(f"Error processing result {url}: {e}")
            summary = snippet
        
        # Store result at the correct index
        all_summaries[index] = summary

    def _process_results_parallel(self, results, query):
        """
        Process multiple search results in parallel using threading.
        
        Args:
            results: List of search result dictionaries
            query: Original search query
        
        Returns:
            List of summaries in the same order as input results
        """
        # Pre-allocate list with None values to maintain order
        all_summaries = [None] * len(results)
        threads = []
        
        # Create and start a thread for each result
        for index, result in enumerate(results):
            thread = threading.Thread(
                target=self._process_single_result,
                args=(result, query, all_summaries, index)
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return all_summaries

    def askInternet(self, query, max_results=5):
        """
        Perform a search using DDGS and process results with summarization
        
        Args:
            query: Search query string
            max_results: Maximum number of results to fetch and process
        
        Returns:
            List of summaries from search results
        """
        # Perform the search https://pypi.org/project/ddgs/
        results = self.search.text(
            query, 
            backend="duckduckgo,bing,yahoo,brave,mojeek,mullvad_brave,mullvad_google,wikipedia,yandex", 
            max_results=max_results
        )
        
        print(results)
        print("=" * 50)
        
        # Show all the results
        for res in results:
            print(res['href'])
        print("=" * 50)
        
        # Process results in parallel
        all_summaries = self._process_results_parallel(results, query)
        
        return all_summaries


if __name__ == "__main__":
    
    ws = WebSearch()
    
    summary = ws.askInternet("latest news headlines")

    print("\n\nDDGS ALL SUMMARIES:\n\n")
    print(summary,"\n\n",len(summary))

    x = ws.summarize_text("".join(summary))
    print("\n\nSUMMARY OF ALL SUMMARIES:\n\n")
    print(x)
