
import pandas as pd
import numpy as np
import faiss
import joblib
import re
from sentence_transformers import SentenceTransformer

class JobRecommender:
    def __init__(self, model_path='./'):
        # 1. Load Artifacts
        self.xgb = joblib.load(f'{model_path}xgb_ranker_model3.pkl')
        self.lgb = joblib.load(f'{model_path}lgb_ranker_model3.pkl')
        self.scaler = joblib.load(f'{model_path}scaler_model3.pkl')
        self.index = faiss.read_index(f'{model_path}job_faiss.index')
        self.df_metadata = pd.read_parquet(f'{model_path}job_metadata_model3.parquet')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define features in exact same order as training
        self.feat_cols = [
            'semantic_score', 'skills_exact_match', 'skills_match_ratio', 
            'n_skills_job', 'n_skills_resume', 'location_match', 
            'exp_diff', 'is_internship', 'is_entry_level', 
            'is_senior_level', 'title_sim', 'desc_len'
        ]

    def _clean_text(self, text):
        if not text: return ""
        return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

    def get_recommendations(self, resume_data, top_n=5):
        # 1. Vectorize Resume Text
        resume_text = f"{resume_data['title']} . {' '.join(resume_data['skills'])} . {resume_data['description']}"
        resume_vec = self.embedder.encode([resume_text]).astype('float32')
        
        # 2. FAISS Retrieval (Top 100 Candidates)
        distances, indices = self.index.search(resume_vec, 100)
        candidates = self.df_metadata.iloc[indices[0]].copy()
        candidates['semantic_score'] = distances[0]
        
        # 3. Dynamic Feature Engineering (Re-ranker Input)
        # Simplified for inference; mirrors the logic from Model 3 training
        features = pd.DataFrame(index=candidates.index)
        features['semantic_score'] = candidates['semantic_score']
        features['skills_exact_match'] = candidates['skills_list'].apply(
            lambda x: len(set(x) & set(resume_data['skills']))
        )
        features['skills_match_ratio'] = features['skills_exact_match'] / (len(resume_data['skills']) + 1e-6)
        features['n_skills_job'] = candidates['skills_list'].apply(len)
        features['n_skills_resume'] = len(resume_data['skills'])
        features['location_match'] = (candidates['location'] == self._clean_text(resume_data['location'])).astype(int)
        features['exp_diff'] = (candidates['exp_years'] - resume_data['exp_years']).abs()
        features['is_internship'] = candidates['title_clean'].str.contains('intern').astype(int)
        features['is_entry_level'] = (candidates['exp_years'] <= 1).astype(int)
        features['is_senior_level'] = (candidates['exp_years'] >= 5).astype(int)
        features['title_sim'] = candidates['title_clean'].apply(
            lambda x: 1.0 if self._clean_text(resume_data['title']) in x else 0.0
        )
        features['desc_len'] = candidates['job_text'].str.len()

        # 4. Scale and Re-rank
        X_scaled = self.scaler.transform(features[self.feat_cols])
        preds = 0.5 * self.xgb.predict(X_scaled) + 0.5 * self.lgb.predict(X_scaled)
        candidates['final_score'] = preds
        
        return candidates.sort_values('final_score', ascending=False).head(top_n)
