
"""
resume_scorer.py — Production wrapper for Model 2
Usage:
    from resume_scorer import ResumeScorer
    from resume_parser import ResumeParser          # Model 1

    parser = ResumeParser('resume_parser_model')
    scorer = ResumeScorer('resume_scorer_model')

    resume_json = parser.parse_pdf('resume.pdf')    # Model 1
    result      = scorer.score(resume_json, jd_text, jd_title, jd_company)
"""
import json, re, pickle, numpy as np, xgboost as xgb
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity

class ResumeScorer:
    def __init__(self, model_dir: str):
        self.cfg = json.load(open(f"{model_dir}/config.json"))
        self.xgb = xgb.XGBRegressor(); self.xgb.load_model(f"{model_dir}/xgb_model.json")
        self.lgb = pickle.load(open(f"{model_dir}/lgb_model.pkl", "rb"))
        self.scaler = pickle.load(open(f"{model_dir}/scaler.pkl", "rb"))
        self.tfidf  = pickle.load(open(f"{model_dir}/tfidf_vectorizer.pkl", "rb"))
        self.ew     = self.cfg["ensemble_weight"]
        self.vocab  = self.cfg["tech_skills_vocab"]
        self.degree_rank = self.cfg["degree_rank"]

    def _clean(self, text):
        text = re.sub(r"<[^>]+>", " ", str(text))
        text = re.sub(r"[\r\n\t]+", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()

    def _extract_skills(self, text):
        t = text.lower()
        return [sk for sk in self.vocab if re.search(r"\b" + re.escape(sk) + r"\b", t)]

    def score(self, resume_json: dict, jd_text: str,
               jd_title: str = "", jd_company: str = "",
               jd_exp_years: float = None) -> dict:
        jd_clean  = self._clean(jd_text)
        jd_skills = self._extract_skills(jd_clean)
        if jd_exp_years is None:
            m = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", jd_clean, re.I)
            jd_exp_years = int(m.group(1)) if m else 2

        r_skills = [s.lower() for s in resume_json.get("skills", [])]
        r_set = set(r_skills)

        def exact_match(jd_sk, r_set):
            if not jd_sk: return 0.5
            return sum(1 for s in jd_sk if s in r_set or any(s in rs for rs in r_set)) / len(jd_sk)

        def fuzzy_match(jd_sk, r_set):
            if not jd_sk or not r_set: return 0.0
            return float(np.mean([max(fuzz.partial_ratio(j, r)/100 for r in r_set) for j in jd_sk]))

        def tfidf_score(rt, jt):
            if not rt or not jt: return 0.0
            v = self.tfidf.transform([rt, jt])
            return float(cosine_similarity(v[0], v[1])[0, 0])

        def exp_match(r_exp, jd_exp):
            d = r_exp - jd_exp
            return min(1.0, 1 - d*0.02) if d >= 0 else max(0.0, 1 + d*0.15)

        def edu_score(deg, jt):
            best = 1
            for d in deg:
                for k, v in self.degree_rank.items():
                    if k in d.lower(): best = max(best, v)
            req = 3
            if any(k in jt.lower() for k in ["phd","doctorate"]): req=5
            elif any(k in jt.lower() for k in ["master","ms ","mba"]): req=4
            d = best - req
            return 1.0 if d >= 0 else max(0.0, 1 + d*0.3)

        r_text = resume_json.get("raw_text", " ".join(r_skills + resume_json.get("designations",[]) + resume_json.get("degree",[])))
        feats = [
            exact_match(jd_skills, r_set),
            fuzzy_match(jd_skills, r_set),
            tfidf_score(r_text, jd_clean),
            exp_match(resume_json.get("years_experience", 0), jd_exp_years),
            edu_score(resume_json.get("degree", []), jd_clean),
            max([fuzz.token_sort_ratio(d.lower(), jd_title.lower())/100 for d in resume_json.get("designations",[''])], default=0.3),
            float(any(any(w in jd_clean.lower() for w in c.lower().split() if len(w)>3) for c in resume_json.get("certifications",[]))),
            sum([0.35*bool(resume_json.get("skills")), 0.20*bool(resume_json.get("degree")),
                 0.20*bool(resume_json.get("companies")), 0.15*bool(resume_json.get("designations")),
                 0.10*bool(resume_json.get("certifications"))])
        ]
        fs = self.scaler.transform(np.array(feats).reshape(1, -1))
        score = float(np.clip(self.ew*self.xgb.predict(fs)[0] + (1-self.ew)*self.lgb.predict(fs)[0], 0, 100))
        missing = [s for s in jd_skills if s not in r_set and not any(fuzz.partial_ratio(s,r)>85 for r in r_set)]
        matched = [s for s in jd_skills if s in r_set or any(fuzz.partial_ratio(s,r)>85 for r in r_set)]
        label = ("Strong match" if score>=75 else "Good match" if score>=55 else "Fair match" if score>=40 else "Weak match")
        return {"ats_score": round(score,1), "label": label,
                "matched_keywords": matched, "missing_keywords": missing,
                "job_title": jd_title, "company": jd_company}
