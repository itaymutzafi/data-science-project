import json
import os

notebook_path = 'Final_Project_Report.ipynb'

def refactor_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Update Status Table
    print("Updating Status Table...")
    for cell in data['cells']:
        if cell.get('id') == 'editing_management':
            source = cell['source']
            new_source = []
            for line in source:
                if "| 3.3 Sentiment Analysis | | | |" in line:
                    new_source.append("| 3.3 Sentiment Analysis | Completed | Pending | Implemented Decay & Market Context | Itay/Moran\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

    # 2. Refactor Cells 46, 47 (Verification & Advanced Viz)
    # We identify them by ID or content
    
    new_cells = []
    
    for cell in data['cells']:
        cell_id = cell.get('id', '')
        source_text = "".join(cell.get('source', []))
        
        # Replace Verification Cell
        if cell_id == 'sentiment_verification_decay' or "Verification of Exponential Decay" in source_text:
            print("Refactoring Verification Cell...")
            cell['source'] = [
                "# Verification of Exponential Decay\n",
                "plot_sentiment_decay_verification()\n"
            ]
            
        # Replace Advanced Viz Cell
        elif cell_id == 'sentiment_advanced_viz' or "# Visualize Advanced Features for Apple" in source_text:
            print("Refactoring Advanced Visualization Cell...")
            cell['source'] = [
                "# Visualize Advanced Features (All Companies)\n",
                "if 'daily_sentiment_all' in locals() and not daily_sentiment_all.empty:\n",
                "    plot_advanced_sentiment_features(daily_sentiment_all)\n",
                "else:\n",
                "    print('No sentiment data generated.')\n"
            ]
            
        # Check for potential "Cell 42" irrelevance
        # If we find a duplicate sentiment visualization cell that isn't the main one
        # ID: sentiment_visualization
        # We want to keep the one that is valid. I removed one before.
        # If user says "clean cell 42", and it's irrelevant.
        # Let's verify if there is an empty cell or a simple print cell we can remove.
        # For now, we mainly focus on the requested refactoring.
        
        new_cells.append(cell)
        
    data['cells'] = new_cells

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print(f"Successfully refactored {notebook_path}")

if __name__ == "__main__":
    refactor_notebook()
