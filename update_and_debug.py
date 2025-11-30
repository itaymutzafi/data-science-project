import json

nb_path = 'Final_Project_Report.ipynb'
debug_script_path = 'debug_notebook.py'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Table
table_cell = nb['cells'][1] # Assuming index 1 as inserted previously
source = table_cell['source']
new_source = []

updates = {
    '| 1.2 Target Variable Definition': '| 1.2 Target Variable Definition | Completed (Shaked) | Pending | Added Note on Inflation |\n',
    '| 2.2 Statistical Properties': '| 2.2 Statistical Properties | Completed (Shaked) | Pending | Added Weak Stationarity Def |\n',
    '| 2.3 Feasibility Analysis': '| 2.3 Feasibility Analysis | Completed (Shaked) | Pending | Added Autocorrelation Plot |\n',
    '| 4.2 Validation Scheme': '| 4.2 Validation Scheme | Completed (Shaked) | Pending | Merged Residual Analysis Note |\n',
    '| 5.1 Baseline Models': '| 5.1 Baseline Models | Completed (Shaked) | Pending | Added CAPM Baseline |\n'
}

for line in source:
    updated_line = line
    for key, value in updates.items():
        if line.startswith(key):
            updated_line = value
            break
    new_source.append(updated_line)

table_cell['source'] = new_source

# Save Notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Table updated successfully.")

# 2. Extract Code
code_content = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code_content.append(f"# Cell ID: {cell.get('id', 'NO_ID')}\n")
        code_content.extend(cell['source'])
        code_content.append('\n\n')

with open(debug_script_path, 'w', encoding='utf-8') as f:
    f.writelines(code_content)

print(f"Code extracted to {debug_script_path}")
