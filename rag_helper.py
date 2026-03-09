import pandas as pd
import numpy as np
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from verification_helper import is_page_verified

# Suppress warnings from Future behavior in pandas/sklearn
warnings.filterwarnings('ignore')

class SafeAdRAG:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Loading Text Embedding Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.safe_ads = []
        self.safe_embeddings = None

    def build_index(self, ads_df, verified_map):
        """
        Extracts captions from ads belonging to verified pages and computes their embeddings.
        """
        safe_texts = []
        for _, row in ads_df.iterrows():
            if is_page_verified(row.get('page_url', ''), verified_map):
                caption = str(row.get('ad_caption', '')).strip()
                if caption and caption.lower() != 'nan' and len(caption) > 10:
                    safe_texts.append(caption)
        
        # Remove exact duplicates
        self.safe_ads = list(set(safe_texts))
        
        if self.safe_ads:
            print(f"Embedding {len(self.safe_ads)} verified safe ads to build RAG index...")
            self.safe_embeddings = self.model.encode(self.safe_ads, show_progress_bar=True)
            print("RAG index built successfully.")
        else:
            print("No verified safe ads found for RAG.")
            self.safe_embeddings = np.array([])

    def get_similar_safe_ads(self, query_text, top_k=2, similarity_threshold=0.60):
        """
        Retrieves top-k similar safe captions.
        """
        if not self.safe_ads or not query_text.strip() or self.safe_embeddings is None or len(self.safe_embeddings) == 0:
            return []
            
        query_emb = self.model.encode([query_text], show_progress_bar=False)
        similarities = cosine_similarity(query_emb, self.safe_embeddings)[0]
        
        # Get top k index sorted by highest similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                results.append(self.safe_ads[idx])
                
        return results
