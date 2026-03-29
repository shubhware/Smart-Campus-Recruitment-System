# 🎓 Smart Campus Recruitment System

An automated, AI-driven pipeline designed to streamline campus placements. This system parses unstructured PDF resumes into structured data using Deep Learning (NER) and evaluates them against specific Job Descriptions using an ensemble of traditional Machine Learning algorithms to generate an actionable ATS (Applicant Tracking System) score.

## 🏗️ Repository Structure

```text
Smart-Campus-Recruitment-System/
│
├── Model1_Resume_Parser/
│   ├── resume_parser_model/     # (Downloaded externally due to size)
│   ├── resume_parser.py         # Inference wrapper
│   ├── requirements.txt         
│   └── resume_parser_model1.ipynb 
│
├── Model2_Resume_Scorer/
│   ├── resume_scorer_model/     # (Downloaded externally)
│   ├── resume_scorer.py         # Inference wrapper
│   ├── requirements.txt
│   └── resume_scorer_model2.ipynb 
│
├── .gitignore
└── README.md
```
## Model 1: Resume Parser (BERT NER)

The first module is a custom Named Entity Recognition (NER) model fine-tuned on `bert-base-uncased`. It extracts structured data (Skills, Degrees, Companies, Locations, etc.) from raw PDF resumes to be passed to downstream scoring models.

### Datasets Used: 
1. Kaggle => Resume Entities for NER by Dataturks (https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner)
2. Kaggle => Updated Resume Dataset by Jillani SofTech (https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset)

### Performance
* **F1 Score:** 0.5278
* **Precision:** 0.5046
* **Recall:** 0.5533

![Training Curves](Model1_Resume_Parser/assets/training_curves.png)

---

## Model 2: ATS Scorer (Machine Learning Ensemble)

The second module acts as the evaluation engine. Because commercial ATS algorithms are proprietary, this model uses a heavily engineered pseudo-labeling approach to simulate industry-standard scoring. It compares the JSON output from Model 1 against a target Job Description.

### Datasets Used: 
1. Kaggle => LinkedIn Job Postings 2023-2024 (https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
2. Kaggle => Resume Entities for NER by Dataturks (https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner)

### Architecture
* A weighted ensemble of **XGBoost** and **LightGBM** regressors.

### Feature Engineering (8 Dimensions)
* **Exact Skills Match** (35% weight)
* **TF-IDF Keyword Coverage** (20% weight)
* **Experience Alignment** (18% weight)
* **Education Level Match** (12% weight)
* **Fuzzy Skills Match** (5% weight)
* **Job Title / Designation Match** (5% weight)
* **Certification Bonus** (3% weight)
* **Resume Completeness** (2% weight)

### Performance (Test Set)
* **Spearman ρ:** 0.8841 (Excellent ranking capability)
* **MAE:** 3.254 (Mean error of only ~3 points out of 100)
* **RMSE:** 4.044

### Output
An overall ATS score (0-100), section-level scores, missing keywords, and an AI-generated list of actionable suggestions to improve the resume.

![Model Performance](Model2_Resume_Scorer/assets/model_performance.png)

---

## 🚀 How to Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/shubhware/Smart-Campus-Recruitment-System.git
cd Smart-Campus-Recruitment-System
```
### Step 2: Download the Pre-Trained Weights
Because the model artifacts contain large weight files (~400MB for BERT), they are hosted securely externally rather than on GitHub.

1. Model 1 (Parser): (https://drive.google.com/file/d/1yhSuiCwkQmNn56UR6bEDx6-BjzNoqnNC/view?usp=share_link)

2. Model 2 (Scorer): (https://drive.google.com/file/d/18UELix1nMT7TZO2pYKWLLNa0e-FVxuI9/view?usp=share_link)

3. Extract both downloaded .zip files.

4. Place the resume_parser_model folder inside the model1_resume_parser/ directory.

5. Place the resume_scorer_model folder inside the model2_resume_scorer/ directory.

### Step 3: Environment Setup

## For the Parser (Model 1):
```bash
cd ../Model1_Resume_Parser
pip install -r requirements.txt
```
## For the Scorer (Model 2):
```bash
cd ../Model2_Resume_Scorer
pip install -r requirements.txt
```
