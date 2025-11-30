import json

nb_path = 'Final_Project_Report.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

table_source = [
    "# Editing Management Table\n",
    "\n",
    "| Section | Editing Status | Reviewed | Notes |\n",
    "| :--- | :--- | :--- | :--- |\n",
    "| **1. Business Understanding** | | | |\n",
    "| 1.1 Project Objective & Goals | | | |\n",
    "| 1.2 Target Variable Definition | | | |\n",
    "| 1.3 Success Criteria & KPIs | | | |\n",
    "| **2. Data Access & EDA** | | | |\n",
    "| 2.1 Data Ingestion & Schema | | | |\n",
    "| 2.2 Statistical Properties | | | |\n",
    "| 2.3 Feasibility Analysis | | | |\n",
    "| **3. Feature Engineering** | | | |\n",
    "| 3.1 Data Cleaning | | | |\n",
    "| 3.2 Technical Indicators | | | |\n",
    "| 3.3 Sentiment Analysis | | | |\n",
    "| 3.4 Feature Selection | | | |\n",
    "| **4. Methodology** | | | |\n",
    "| 4.1 Evaluation Metrics | | | |\n",
    "| 4.2 Validation Scheme | | | |\n",
    "| **5. Modeling** | | | |\n",
    "| 5.1 Baseline Models | | | |\n",
    "| 5.2 Linear Models | | | |\n",
    "| 5.3 Advanced Models | | | |\n",
    "| **6. Evaluation** | | | |\n",
    "| 6.1 Error Analysis | | | |\n",
    "| 6.2 Trading Strategy | | | |\n",
    "| 6.3 Comparison to Baselines | | | |\n",
    "| **7. Explainability** | | | |\n",
    "| 7.1 Feature Importance | | | |\n",
    "| 7.2 SHAP Values | | | |\n",
    "| **8. Conclusion** | | | |\n",
    "| 8.1 Summary of Findings | | | |\n",
    "| 8.2 Proposed Improvements | | | |\n"
]

new_cell = {
    "cell_type": "markdown",
    "id": "editing_management",
    "metadata": {},
    "source": table_source
}

# Insert at index 1 (after title, before TOC)
nb['cells'].insert(1, new_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Editing Management Table inserted successfully.")
