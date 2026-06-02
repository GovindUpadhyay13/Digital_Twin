import os
import re
import requests
import yaml
from bs4 import BeautifulSoup
from pathlib import Path

class BlogCollector:
    def __init__(self, output_dir="data/raw/blog"):
        self.output_dir = output_dir
        base_dir = Path(__file__).parent
        config_path = base_dir / "config_sources.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f).get("blog", {})
        else:
            self.config = {}

    def collect(self):
        """Scrape both karpathy.github.io and fall back to mock blog posts if needed"""
        os.makedirs(self.output_dir, exist_ok=True)
        github_io_url = self.config.get("github_io_url", "https://karpathy.github.io")
        
        success = False
        try:
            print(f"Scraping Karpathy's Github Blog at {github_io_url}...")
            response = requests.get(github_io_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Find all blog post links (usually formatted like /2015/05/21/rnn-effectiveness/ or similar)
                links = soup.find_all('a', href=re.compile(r'/\d{4}/\d{2}/\d{2}/'))
                
                if links:
                    for link in links:
                        post_href = link['href']
                        title = link.get_text().strip()
                        post_url = post_href if post_href.startswith("http") else github_io_url + post_href
                        
                        try:
                            post_res = requests.get(post_url, timeout=10)
                            if post_res.status_code == 200:
                                post_soup = BeautifulSoup(post_res.text, 'html.parser')
                                
                                # Extract content: usually inside article tag, post class, or entry-content class
                                content_node = post_soup.find('article') or post_soup.find(class_='post') or post_soup.find(class_='entry-content')
                                content = content_node.get_text() if content_node else post_soup.get_text()
                                
                                # Clean content a bit
                                content = re.sub(r'\n+', '\n', content).strip()
                                
                                # Extrapolate date from URL
                                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', post_url)
                                date_str = "-".join(date_match.groups()) if date_match else "2015-01-01"
                                
                                self._save_post(title, content, post_url, "github.io", date_str)
                                success = True
                        except Exception as e:
                            print(f"[WARN] Failed to scrap individual blog post {title}: {e}")
        except Exception as e:
            print(f"[WARN] Blog Scraping failed: {e}")
            
        if not success:
            print("[WARN] Scraper was unable to collect blog posts. Loading mock blog posts from config_sources.yaml...")
            mock_posts = self.config.get("mock_posts", [])
            for post in mock_posts:
                self._save_post(
                    title=post.get("title", "Untitled Post"),
                    content=post.get("content", ""),
                    url=post.get("url", "https://karpathy.github.io"),
                    platform="github.io",
                    date=post.get("date", "2015-01-01")
                )

    def _save_post(self, title, content, url, platform, date):
        """Save post to output directory"""
        filename = f"{self.output_dir}/{title.replace(' ', '_')[:50]}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n")
            f.write(f"Date: {date}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Platform: {platform}\n\n")
            f.write(content)
        print(f"[OK] Saved blog post: {title}")

if __name__ == "__main__":
    BlogCollector().collect()
