
"""
skill_gap_analyzer.py — Production wrapper for Model 4

Usage:
    from resume_parser      import ResumeParser       # Model 1
    from resume_scorer      import ResumeScorer       # Model 2
    from job_recommender    import JobRecommender     # Model 3
    from skill_gap_analyzer import SkillGapAnalyzer  # Model 4

    parser      = ResumeParser("resume_parser_model")
    scorer      = ResumeScorer("resume_scorer_model")
    recommender = JobRecommender("job_recommender_model")
    analyzer    = SkillGapAnalyzer("skill_gap_model")

    m1   = parser.parse_pdf("resume.pdf")
    m2   = scorer.score(m1, jd_text, jd_title)
    m3   = recommender.recommend(m1, m2, top_k=20)
    report = analyzer.analyze(m1, m2, m3)
"""
import json, re, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import networkx as nx
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

class SkillGapAnalyzer:
    def __init__(self, model_dir: str):
        self.cfg = json.load(open(f"{model_dir}/config.json"))
        self.xgb_cls  = xgb.XGBClassifier();  self.xgb_cls.load_model(f"{model_dir}/xgb_trend_classifier.json")
        self.xgb_pri  = xgb.XGBRegressor();   self.xgb_pri.load_model(f"{model_dir}/xgb_priority_scorer.json")
        self.lgb_cls  = pickle.load(open(f"{model_dir}/lgb_trend_classifier.pkl","rb"))
        self.sc_tr    = pickle.load(open(f"{model_dir}/scaler_trend.pkl","rb"))
        self.sc_pr    = pickle.load(open(f"{model_dir}/scaler_priority.pkl","rb"))
        self.tfidf    = pickle.load(open(f"{model_dir}/tfidf_courses.pkl","rb"))
        self.courses  = pd.read_parquet(f"{model_dir}/course_catalog.parquet")
        self.trends   = pd.read_parquet(f"{model_dir}/skill_trends.parquet")
        self.role_dem = json.load(open(f"{model_dir}/role_skill_demand.json"))
        self.course_mat = self.tfidf.transform(self.courses["text_clean"])
        self.ew       = self.cfg["ens_weight"]
        self.tmap     = self.cfg["trend_map"]
        self.tsmap    = self.cfg["trend_score_map"]
        self.lw       = self.cfg["learn_weeks"]
        # Build graph
        self.G = nx.DiGraph()
        for sk, prereqs in self.cfg["skill_prereqs"].items():
            self.G.add_node(sk)
            for p in prereqs: self.G.add_edge(p, sk)

    def _predict_trend(self, feat_row):
        X = self.sc_tr.transform(np.array(feat_row).reshape(1,-1))
        p_xgb = self.xgb_cls.predict_proba(X)[0]
        p_lgb = self.lgb_cls.predict_proba(X)[0]
        return int(np.argmax(self.ew*p_xgb + (1-self.ew)*p_lgb))

    def _courses_for(self, skill, k=3):
        qv = self.tfidf.transform([skill.lower()])
        sims = cosine_similarity(qv, self.course_mat)[0]
        top = np.argsort(sims)[::-1][:k*3]
        out = []
        for i in top:
            if sims[i] < 0.01: continue
            r = self.courses.iloc[i]
            out.append({"title":r["title"],"platform":r["platform"],
                         "rating":float(r["rating"]),"is_free":bool(r["is_free"]),"url":r["url"]})
        out.sort(key=lambda x: -x["rating"])
        return out[:k]

    def analyze(self, m1, m2, m3, target_role=None, top_k=8):
        current = set(s.lower() for s in m1.get("skills",[]))
        recs    = m3.get("recommendations",[])
        if not target_role:
            cats = [r.get("category","") for r in recs[:5]]
            target_role = Counter(cats).most_common(1)[0][0] if cats else "Data Science"
        gap_ct = Counter()
        for sk in m2.get("missing_keywords",[]): gap_ct[sk.lower()] += 3
        for r in recs[:10]:
            for sk in r.get("skill_gaps",[]): gap_ct[sk.lower()] += 2
        role_dem = self.role_dem.get(target_role, self.role_dem.get("Data Science",{}))
        for sk, d in role_dem.items():
            if sk not in current and d > 5: gap_ct[sk] += max(1, int(d//10))
        for sk in list(gap_ct):
            if sk in current: del gap_ct[sk]
        gaps = []
        id2l = {v:k for k,v in self.tmap.items()}
        for sk, _ in gap_ct.most_common(top_k*2):
            row = self.trends[self.trends["skill"]==sk]
            if not row.empty:
                tl  = row.iloc[0]["trend_label"]
                yoy = float(row.iloc[0]["yoy_growth_mean"])
                pct = row.iloc[0]["pct_latest"]
            else:
                tl = id2l.get(0, "stable"); yoy = 0.0; pct = 0.0
            ts  = self.tsmap[tl]
            dem = role_dem.get(sk, 0.0)
            P   = self.sc_pr.transform(np.array([[dem,ts,np.clip(yoy,-5,10),1.0,min(100,dem*1.2)]]))
            pri = float(np.clip(self.xgb_pri.predict(P)[0],0,100))
            urg = "critical" if pri>=75 else "high" if pri>=55 else "medium" if pri>=35 else "low"
            gaps.append({"skill":sk,"urgency":urg,"trend":tl,"demand_pct":round(dem,1),
                          "yoy_growth_pct":round(yoy,2),"priority_score":round(pri,1),
                          "learning_weeks":self.lw.get(sk,2),"courses":self._courses_for(sk)})
        gaps.sort(key=lambda x:-x["priority_score"])
        gaps = gaps[:top_k]
        ordered = []
        sub = set(g["skill"] for g in gaps)
        for sk in sub:
            for p in self.cfg["skill_prereqs"].get(sk,[]):
                if p not in current: sub.add(p)
        sg = self.G.subgraph(sub).copy()
        try: ordered = [s for s in nx.topological_sort(sg) if s not in current]
        except: ordered = list(sub - current)
        roadmap, wk = [], 1
        for ph, sk in enumerate(ordered[:10], 1):
            wn = self.lw.get(sk, 2)
            c  = self._courses_for(sk, 1)
            roadmap.append({"phase":ph,"weeks":f"{wk}–{wk+wn-1}","skill":sk,
                             "milestone":f"Complete {sk} fundamentals",
                             "resource":c[0]["title"] if c else f"Search {sk} tutorial"})
            wk += wn
        rising = self.trends[self.trends["trend_label"]=="rising"].nlargest(20,"rel_growth")["skill"].tolist()
        return {"target_role":target_role,"current_skills":list(current),
                "overall_gap_score":round(float(np.mean([g["priority_score"] for g in gaps])) if gaps else 0,1),
                "estimated_upskill_weeks":wk-1,"skill_gaps":gaps,
                "rising_skills_to_watch":[s for s in rising if s not in current][:6],
                "roadmap":roadmap}
