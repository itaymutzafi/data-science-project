# Project Standards & Workflow Guidelines

This document outlines the agreed-upon standards for our Data Science project to ensure consistency, reproducibility, and high-quality academic output.

## 1. Notebook Workflow & Structure

To keep notebooks clean, readable, and focused on analysis:

*   **Imports**: All imports must be placed in the very first cell of the notebook. No hidden imports in the middle of the analysis.
*   **Cell Structure**:
    *   Break down logic into manageable cells. Avoid long, monolithic scripts.
    *   **Narrative First**: Every code cell should be preceded by a markdown cell explaining *what* is being done and *why*. This constitutes the "discussion" part of our academic standard.
    *   **Status Tracking**:
        *   Maintain the "Editing Management Table" at the top of the notebook.
        *   Mark sections as "Done" only when the code is complete, tested, and documented.
    *   **Table of Contents**:
        *   Update the Table of Contents (TOC) whenever sections are added, removed, or renamed.
        *   Ensure the TOC links correctly point to the corresponding sections.
*   **Code Usage**:
    *   Notebooks should primarily contain *calls* to core logic defined in `src/`.
    *   Avoid defining complex functions or classes directly in the notebook unless they are strictly for one-off visualization or experimentation.
    *   This reduces visual clutter and ensures logic is reusable and testable.

## 2. Code Quality & Structure (`src/`)

*   **Tight Code**: Write concise, focused functions.
*   **No Duplication**:
    *   Do not duplicate imports across files if not needed (though each file needs its own imports).
    *   Refactor common logic into shared utilities.
*   **Core Logic**: All heavy lifting (data processing, model training, complex plotting) goes into `src/`.
*   **Docstrings**: Every function and module must have a docstring explaining its purpose, arguments, and return values.

## 3. Data Management

*   **Single Source of Truth**: All data resides in the root `data/` directory.
    *   `data/raw/`: Immutable original data.
    *   `data/processed/`: Cleaned and transformed data.
*   **Git Ignore**: The `data/` directory is strictly ignored by git (except for `.gitkeep` files) to prevent bloating the repository.
*   **Loader Paths**: All data loading functions in `src/data/loader.py` must point to the root `data/` directory relative to the project root, not a local `src/data/` folder.

## 4. Visualization Standards

*   **Uniformity**: All plots must use the style defined in `src.evaluation.plots`.
*   **Academic Quality**:
    *   Figures should be high-resolution (DPI=150+).
    *   Labels, titles, and legends must be clear and legible.
    *   Use the `set_style()` function at the start of notebooks.

## 5. Git Workflow

*   **Clean Commits**: Avoid committing large files or unnecessary metadata.
*   **Pull Before Push**: Always pull the latest changes before pushing to avoid conflicts.
*   **Review**: Briefly review your code against these standards before committing.
