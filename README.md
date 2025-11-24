# Data Science Workshop Project - <Insert Project Title Here>

This repository contains the full workflow of our group project in the Data Science course at Tel Aviv University.

The project topic will be finalized and approved in class during the upcoming week.  
Once confirmed, all scientific background, research questions, and methodological details will be inserted into this document and the relevant project folders.

---

## 1. Project Overview

### 1.1 Topic (to be completed)
<Insert a short description of the selected topic. For example: analysis of financial time series, market behavior, or prediction models for stock dynamics.>

### 1.2 Motivation (to be completed)
<Explain why this topic is relevant, interesting, and suitable for a full data science workflow.>

### 1.3 Research Questions (to be refined)
Possible structure for later use:
- Question 1: <Insert>
- Question 2: <Insert>
- Question 3: <Insert>

The final research questions will be defined once the dataset and scope are confirmed.

---

## 2. Repository Structure

- **data/**  
  Raw and processed datasets (to be added once the project topic is approved).  
  - `raw/` — original data files  
  - `processed/` — cleaned or transformed versions  
  - `metadata/` — data dictionaries, variable documentation  

- **notebooks/**  
  Jupyter notebooks used during development.  
  - `exploration/` — initial EDA  
  - `modeling/` — feature engineering, model experiments  
  - `main.ipynb` — final linear project workflow  

- **src/**  
  Python modules implementing the reusable parts of the analysis pipeline  
  (loading, cleaning, modeling, evaluation).

- **reports/**  
  Presentations and written materials.  
  - Draft slides  
  - Final presentation  
  - Written summary or short paper  

- **literature/**  
  Relevant course materials and external references.

- **results/**  
  Figures, tables, and evaluation outputs.

---

## 3. Planned Methodology

Once the topic is approved, the project will follow the standard data science pipeline:

1. Data collection and documentation  
2. Exploratory data analysis  
3. Preprocessing and feature engineering  
4. Model development  
5. Evaluation and validation  
6. Interpretation and communication of results  

Each part will be added to the repository in the appropriate folders.

---

## 4. Environment Setup

### Conda environment

```bash
conda env create -f environment.yml
conda activate ds-project
