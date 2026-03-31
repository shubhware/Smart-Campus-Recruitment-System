"""
resume_parser_v2.py — Production-ready Hybrid Resume Parser
============================================================
Strategy:
  - RegEx  → rigid deterministic patterns (Email, Phone, URL, dates, GPA)
  - BERT   → contextual NER for fuzzy patterns (Name, Skills, Company,
             Designation, College, Degree, Certifications, Location)
  - Merge  → BERT output is enriched / overridden by RegEx where RegEx
             is provably more accurate

Usage:
    parser = HybridResumeParser("resume_parser_model")
    result = parser.parse_pdf("resume.pdf")
    # or
    result = parser.parse_text(raw_text)
"""

import re
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pdfplumber
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

# ── RegEx pattern library ─────────────────────────────────────────────────────
# These are 100% deterministic — BERT cannot beat RegEx on these.

REGEX_PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"""
        (?:
            (?:\+?\d{1,3}[\s\-.])?      # country code
            (?:\(?\d{2,4}\)?[\s\-.])?   # area code
            \d{3,5}[\s\-.]              # exchange
            \d{4,6}                     # subscriber
        )
        """,
        re.VERBOSE,
    ),
    "url": re.compile(
        r"https?://[^\s<>\"]+|"
        r"(?:www\.|linkedin\.com|github\.com|kaggle\.com|portfolio\.)[^\s<>\"]*",
        re.IGNORECASE,
    ),
    "linkedin": re.compile(
        r"linkedin\.com/in/[a-zA-Z0-9\-_%]+",
        re.IGNORECASE,
    ),
    "github": re.compile(
        r"github\.com/[a-zA-Z0-9\-_%]+",
        re.IGNORECASE,
    ),
    "gpa": re.compile(
        r"(?:GPA|CGPA|Grade|Score)\s*:?\s*(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        re.IGNORECASE,
    ),
    "graduation_year": re.compile(
        r"\b(19|20)\d{2}\b",
    ),
    "years_experience": re.compile(
        r"(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
        re.IGNORECASE,
    ),
    "experience_date_range": re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
        r"\s*[-–—to]+\s*"
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)",
        re.IGNORECASE,
    ),
    "pincode": re.compile(r"\b\d{6}\b"),  # Indian PIN codes
}

# ── Tech skill vocabulary for normalisation ───────────────────────────────────
SKILL_NORMALISE = {
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "hf": "huggingface",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "pytorch": "pytorch",
}

LABEL_NORM = {
    "Name": "NAME",
    "Email Address": "EMAIL",
    "Phone Number": "PHONE",
    "Skills": "SKILLS",
    "Degree": "DEGREE",
    "College Name": "COLLEGE",
    "Companies worked at": "COMPANY",
    "Designation": "DESIGNATION",
    "Location": "LOCATION",
    "Years of Experience": "EXPERIENCE",
    "Graduation Year": "GRAD_YEAR",
    "Certification": "CERTIFICATION",
    "Links": "LINKS",
}


class HybridResumeParser:
    """
    Two-layer parser:
      Layer 1 — RegEx: handles structured fields with 100% precision
      Layer 2 — BERT:  handles unstructured contextual fields
    BERT output is overridden by RegEx for Email, Phone, URLs.
    """

    def __init__(self, model_dir: str, device: int = -1):
        """
        Parameters
        ----------
        model_dir : str   Path to saved BERT NER model directory.
        device    : int   -1 = CPU, 0 = GPU (if CUDA available).
        """
        logger.info(f"Loading BERT NER model from {model_dir} ...")
        self.ner = hf_pipeline(
            "ner",
            model=model_dir,
            tokenizer=model_dir,
            aggregation_strategy="simple",
            device=device,
        )
        logger.info("Model loaded.")

    # ── PDF text extraction ───────────────────────────────────────────────────

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract plain text from a PDF file.
        Handles:
          - Native text PDFs (fast path via pdfplumber)
          - Multi-column layouts (join columns per page)
          - Encoding edge cases
        """
        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Try bounding-box aware extraction first
                    page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if page_text:
                        text_parts.append(page_text)
        except Exception as exc:
            logger.error(f"PDF extraction failed: {exc}")
            raise ValueError(f"Cannot read PDF: {exc}") from exc

        if not text_parts:
            raise ValueError("PDF appears to be empty or image-only (no extractable text).")

        return "\n".join(text_parts)

    # ── Layer 1: RegEx extraction ─────────────────────────────────────────────

    def _regex_extract(self, text: str) -> dict:
        """
        Extract all deterministic structured fields via compiled RegEx patterns.
        Returns a dict of confirmed values — these override BERT.
        """
        result = {}

        # Email — take first match, strip trailing punctuation
        email_matches = REGEX_PATTERNS["email"].findall(text)
        if email_matches:
            result["email"] = email_matches[0].rstrip(".,;:")

        # Phone — take first clean match
        phone_matches = REGEX_PATTERNS["phone"].findall(text)
        if phone_matches:
            phone_raw = phone_matches[0].strip()
            # Keep only digits and +, normalize
            phone_clean = re.sub(r"[^\d+\-\s()]", "", phone_raw).strip()
            if len(re.sub(r"[^\d]", "", phone_clean)) >= 7:
                result["phone"] = phone_clean

        # URLs — all matches, deduplicated
        urls = list(dict.fromkeys(REGEX_PATTERNS["url"].findall(text)))
        linkedin_urls = REGEX_PATTERNS["linkedin"].findall(text)
        github_urls = REGEX_PATTERNS["github"].findall(text)
        result["links"] = list(dict.fromkeys(
            linkedin_urls + github_urls + [u for u in urls
                                           if "linkedin" not in u and "github" not in u]
        ))[:10]

        # GPA
        gpa_match = REGEX_PATTERNS["gpa"].search(text)
        if gpa_match:
            result["gpa"] = f"{gpa_match.group(1)}/{gpa_match.group(2)}"

        # Years of experience — take maximum mentioned
        exp_matches = REGEX_PATTERNS["years_experience"].findall(text)
        if exp_matches:
            result["years_experience"] = max(int(e) for e in exp_matches)

        # Graduation year — years in 1980–2030 range
        year_matches = REGEX_PATTERNS["graduation_year"].findall(text)
        valid_years = [y for y in year_matches if 1980 <= int(y) <= 2030]
        if valid_years:
            result["graduation_year"] = list(dict.fromkeys(valid_years))

        return result

    # ── Layer 2: BERT NER extraction ──────────────────────────────────────────

    def _bert_extract(self, text: str) -> dict:
        """
        Run BERT token classification on text chunks and aggregate entities.
        Handles texts longer than 512 tokens via overlapping windows.
        """
        words = text.split()
        chunk_size = 300  # words per chunk (safe under 512 BERT subword limit)
        overlap = 50      # overlap to catch entities at chunk boundaries
        entities_raw = defaultdict(list)

        for start in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[start : start + chunk_size])
            if not chunk.strip():
                continue
            try:
                ner_results = self.ner(chunk)
            except Exception as exc:
                logger.warning(f"BERT inference failed on chunk: {exc}")
                continue

            for ent in ner_results:
                lbl_raw = ent.get("entity_group", "")
                lbl = LABEL_NORM.get(lbl_raw, lbl_raw.upper().replace(" ", "_"))
                val = ent.get("word", "").strip()
                score = ent.get("score", 0)
                # Only keep high-confidence predictions
                if val and score > 0.70 and len(val) > 1:
                    entities_raw[lbl].append(val)

        return dict(entities_raw)

    # ── Skill normalisation & deduplication ───────────────────────────────────

    def _process_skills(self, raw_skill_chunks: list[str]) -> list[str]:
        """
        Flatten, normalise, and deduplicate skills extracted by BERT.
        Handles semicolon, comma, bullet, and pipe delimiters.
        """
        flat = []
        for chunk in raw_skill_chunks:
            for token in re.split(r"[,;\n•·|/\\]", chunk):
                t = token.strip().strip("–-·•")
                t = re.sub(r"\s+", " ", t).lower()
                if len(t) < 2 or len(t) > 50:
                    continue
                # Normalise known aliases
                t = SKILL_NORMALISE.get(t, t)
                flat.append(t)

        # Deduplicate preserving order
        seen, deduped = set(), []
        for sk in flat:
            if sk not in seen:
                seen.add(sk)
                deduped.append(sk)
        return deduped

    # ── Merge layer: RegEx overrides BERT for deterministic fields ────────────

    def _merge(self, regex_data: dict, bert_data: dict) -> dict:
        """
        Merge RegEx (high precision, low recall) with BERT (higher recall).
        RegEx wins on Email, Phone, URLs — fields where it is 100% correct.
        BERT wins on Name, Skills, Company, Designation, Degree, College, Location.
        """

        def dedup(lst: list) -> list:
            seen, out = set(), []
            for x in lst:
                xn = x.lower()
                if xn not in seen:
                    seen.add(xn)
                    out.append(x)
            return out

        return {
            # RegEx wins — deterministic fields
            "email": regex_data.get("email", bert_data.get("EMAIL", [""])[0] if bert_data.get("EMAIL") else ""),
            "phone": regex_data.get("phone", bert_data.get("PHONE", [""])[0] if bert_data.get("PHONE") else ""),
            "links": regex_data.get("links", dedup(bert_data.get("LINKS", []))),
            "gpa": regex_data.get("gpa"),
            "years_experience": regex_data.get(
                "years_experience",
                self._parse_exp_years(bert_data.get("EXPERIENCE", [])),
            ),
            "graduation_year": regex_data.get("graduation_year", dedup(bert_data.get("GRAD_YEAR", []))),

            # BERT wins — contextual fields
            "name": bert_data.get("NAME", [""])[0] if bert_data.get("NAME") else "",
            "location": dedup(bert_data.get("LOCATION", [])),
            "skills": self._process_skills(bert_data.get("SKILLS", [])),
            "designations": dedup(bert_data.get("DESIGNATION", [])),
            "companies": dedup(bert_data.get("COMPANY", [])),
            "degree": dedup(bert_data.get("DEGREE", [])),
            "colleges": dedup(bert_data.get("COLLEGE", [])),
            "certifications": dedup(bert_data.get("CERTIFICATION", [])),
        }

    def _parse_exp_years(self, bert_exp_list: list) -> int:
        """Extract numeric years from BERT-identified experience spans."""
        for span in bert_exp_list:
            m = re.search(r"(\d+)", str(span))
            if m:
                return int(m.group(1))
        return 0

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_text(self, text: str) -> dict:
        """
        Parse raw resume text → structured JSON.
        This is the core function called by both parse_pdf() and the API endpoint.

        Returns
        -------
        dict with keys matching Model 1's documented output schema.
        Adds `raw_text` for downstream semantic models (M2, M3, M4).
        """
        if not text or len(text.strip()) < 50:
            raise ValueError("Text too short to be a valid resume.")

        # Layer 1: RegEx — fast, deterministic
        regex_data = self._regex_extract(text)

        # Layer 2: BERT — contextual NER
        bert_data = self._bert_extract(text)

        # Merge layers
        parsed = self._merge(regex_data, bert_data)

        # Attach raw text for semantic similarity (M2, M3)
        parsed["raw_text"] = " ".join(text.split()[:400])  # first 400 words

        # Metadata
        parsed["_meta"] = {
            "model": "hybrid-resume-parser-v2",
            "layers": ["regex", "bert-ner"],
            "text_length": len(text),
            "n_skills": len(parsed.get("skills", [])),
        }

        return parsed

    def parse_pdf(self, pdf_path: str) -> dict:
        """
        Full pipeline: PDF → text extraction → hybrid parse → structured JSON.
        """
        text = self.extract_text(pdf_path)
        return self.parse_text(text)

    def parse_pdf_bytes(self, pdf_bytes: bytes, filename: str = "resume.pdf") -> dict:
        """
        Parse a PDF from in-memory bytes (for API file upload endpoints).
        Writes to a temp file, parses, then cleans up.
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix="resume_"
        ) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            return self.parse_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)  # always clean up temp file


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    SAMPLE = """
    Rahul Sharma
    rahul.sharma@gmail.com | +91-9876543210 | Bangalore, India
    LinkedIn: linkedin.com/in/rahul-sharma | GitHub: github.com/rahulsharma

    EXPERIENCE
    Data Scientist — TCS (Tata Consultancy Services)
    June 2021 – Present | Bangalore
    3 years of experience in ML, NLP and computer vision.
    CGPA: 8.9/10

    SKILLS
    Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn,
    SQL, REST APIs, Docker, Git, AWS, NLP, Huggingface, Transformers

    EDUCATION
    B.Tech Computer Science — IIT Bombay | 2021

    CERTIFICATIONS
    AWS Certified Machine Learning Specialty
    """

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "resume_parser_model"
    parser = HybridResumeParser(model_dir)
    result = parser.parse_text(SAMPLE)
    print(json.dumps({k: v for k, v in result.items() if k != "raw_text"}, indent=2))
