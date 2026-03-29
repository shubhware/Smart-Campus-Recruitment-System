
"""
resume_parser.py  — Production inference wrapper for Model 1
Usage:
    from resume_parser import ResumeParser
    parser = ResumeParser('path/to/saved_model')
    result = parser.parse_pdf('resume.pdf')
"""
import json, re
from collections import defaultdict
import pdfplumber
from transformers import pipeline as hf_pipeline


class ResumeParser:
    def __init__(self, model_dir: str, device: int = -1):
        self.ner = hf_pipeline(
            'ner', model=model_dir, tokenizer=model_dir,
            aggregation_strategy='simple', device=device
        )

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()

    def parse_text(self, text: str) -> dict:
        words = text.split()
        chunks, size, overlap = [], 400, 50
        for start in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[start:start + size]))
            if start + size >= len(words): break

        entities_raw = defaultdict(list)
        for chunk in chunks:
            for ent in self.ner(chunk):
                lbl = ent["entity_group"].upper().replace(" ", "_")
                if ent["score"] > 0.70:
                    entities_raw[lbl].append(ent["word"].strip())

        def dedup(lst):
            seen, out = set(), []
            for x in lst:
                if x.lower() not in seen: seen.add(x.lower()); out.append(x)
            return out

        return {
            "name": entities_raw.get("NAME", [""])[0],
            "email": entities_raw.get("EMAIL", [""])[0],
            "phone": entities_raw.get("PHONE", [""])[0],
            "location": dedup(entities_raw.get("LOCATION", [])),
            "skills": dedup(entities_raw.get("SKILLS", [])),
            "designations": dedup(entities_raw.get("DESIGNATION", [])),
            "companies": dedup(entities_raw.get("COMPANY", [])),
            "degree": dedup(entities_raw.get("DEGREE", [])),
            "colleges": dedup(entities_raw.get("COLLEGE", [])),
            "graduation_year": dedup(entities_raw.get("GRAD_YEAR", [])),
            "years_experience": entities_raw.get("EXPERIENCE", [""])[0],
            "certifications": dedup(entities_raw.get("CERTIFICATION", [])),
        }

    def parse_pdf(self, pdf_path: str) -> dict:
        text = self.extract_text(pdf_path)
        return self.parse_text(text)
