# 🎓 Smart Campus Recruitment System

This repository contains the backend machine learning modules for an automated campus placement system. 

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

## 🚀 How to Run Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/shubhware/Smart-Campus-Recruitment-System.git
cd Smart-Campus-Recruitment-System/Model1_Resume_Parser
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Download the Pre-Trained Model Weights
1. Download the weights here: https://drive.google.com/file/d/1yhSuiCwkQmNn56UR6bEDx6-BjzNoqnNC/view?usp=share_link
2. Extract the .zip file.
3. Move the extracted resume_parser_model folder directly into the Model1_Resume_Parser/ directory.

Your folder structure must look like this before running: Model1_Resume_Parser/resume_parser_model/model.safetensors