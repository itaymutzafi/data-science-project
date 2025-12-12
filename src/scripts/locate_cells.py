import json

notebook_path = 'Final_Project_Report.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

cells = data['cells']

targets = ['sentiment_verification_decay', 'sentiment_advanced_viz', 'sentiment_visualization']

print("Searching for cells by ID and content...")

for i, cell in enumerate(cells):
    cell_id = cell.get('id', 'N/A')
    source = "".join(cell.get('source', []))
    
    # Check if it matches our known IDs or has imports
    if cell_id in targets or "import" in source:
        print(f"Index: {i}, ID: {cell_id}, Type: {cell.get('cell_type')}")
        print(f"Source snippet: {source[:100]}...")
        if "import" in source:
             print("[Contains Imports]")
        print("-" * 20)
