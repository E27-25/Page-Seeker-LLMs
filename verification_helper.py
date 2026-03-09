import pandas as pd
import re

def build_verification_map(feed_csv_path):
    """
    Reads the feed CSV and builds a dictionary mapping page IDs to their verification status.
    """
    try:
        df_feed = pd.read_csv(feed_csv_path)
        verified_map = {}
        for idx, row in df_feed.iterrows():
            is_verified = row.get('feed_is_verified', False)
            if is_verified:
                urls = row.get('feed_profile_urls', '')
                if pd.notna(urls):
                    try:
                        url_list = eval(urls) if isinstance(urls, str) else urls
                        for u in url_list:
                            page_id = extract_page_id(u)
                            if page_id:
                                verified_map[page_id] = True
                    except:
                        pass
        return verified_map
    except Exception as e:
        print(f"Error loading feed data: {e}")
        return {}

def extract_page_id(url):
    """
    Extracts the unique page ID or username from a Facebook URL.
    """
    if pd.isna(url): return ''
    url_str = str(url)
    match = re.search(r'facebook\.com/(?:profile\.php\?id=)?([^/?#]+)', url_str)
    if match: return match.group(1).strip('/')
    return url_str

def is_page_verified(page_url, verified_map):
    """
    Checks if a page URL exists in the verified map.
    """
    page_id = extract_page_id(page_url)
    return verified_map.get(page_id, False)
