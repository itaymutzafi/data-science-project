import json

notebook_path = 'Final_Project_Report.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

cells = data['cells']
start = 38
end = 52

print(f"Total cells: {len(cells)}")

for i in range(start, min(end, len(cells))):
    cell = cells[i]
    cell_type = cell.get('cell_type')
    source = "".join(cell.get('source', []))[:100] + "..." # truncate
    print(f"--- Cell {i} ({cell_type}) ---")
    print(source)
    print("-------------------------")
