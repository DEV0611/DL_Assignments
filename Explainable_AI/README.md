# Explainable AI (XAI) Project

A comprehensive XAI project applying interpretable machine learning techniques to **tabular clinical data** and **image classification**, demonstrating global and local explanation methods across modalities.

## Project Structure

```
Explainable-AI/
├── notebooks/
│   ├── 01_tabular_xai.ipynb      # Heart failure prediction + SHAP + LIME
│   ├── 02_cnn_gradcam.ipynb      # Flower classification + Grad-CAM
│   └── 03_xai_analysis.ipynb     # Cross-method comparison & synthesis
├── outputs/                       # Generated plots and figures
├── src/                           # Reusable utility scripts
├── data/                          # Downloaded datasets (auto-created)
├── REPORT.md                      # Full project report
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

```bash
# Open Jupyter
jupyter notebook notebooks/
```

Run notebooks sequentially:
1. `01_tabular_xai.ipynb` — Tabular model with SHAP/LIME
2. `02_cnn_gradcam.ipynb` — CNN with Grad-CAM
3. `03_xai_analysis.ipynb` — Comparative analysis

### 3. View Outputs

All visualizations are saved to `outputs/` directory with numbered prefixes:
- `01-04`: Target distribution, correlations, model comparison, confusion matrices
- `05-09`: SHAP summary, bar, dependence, waterfall plots
- `10-11`: LIME explanations
- `12-13`: Cross-model importance, bias analysis
- `14-23`: Flower dataset, training history, CNN metrics, Grad-CAM visualizations
- `24`: XAI technique comparison

## Datasets

| Dataset | Source | Size | Task |
|---------|--------|------|------|
| Heart Failure Clinical Records | UCI ML Repository | 299 patients, 12 features | Binary classification |
| TensorFlow Flowers | TensorFlow datasets | ~3,670 images, 5 classes | Multi-class classification |

Both datasets are automatically downloaded when running the notebooks.

## XAI Techniques Used

### Tabular (Notebook 01)
- **SHAP Summary Plot** — Global feature importance with directionality
- **SHAP Bar Plot** — Ranked mean |SHAP| values
- **SHAP Dependence Plots** — Feature effect visualization
- **SHAP Waterfall** — Local explanation for individual predictions
- **LIME Tabular** — Local surrogate model explanations
- **Cross-model feature importance** — Comparing XGBoost, RF, LR

### CNN (Notebook 02)
- **Grad-CAM** — Gradient-weighted class activation maps
- **Misclassification Grad-CAM** — Understanding why the model errs
- **Filter Visualization** — What each convolutional filter detects

### Analysis (Notebook 03)
- **SHAP vs LIME agreement** — Cross-validation of explanations
- **Medical literature validation** — Clinical verification of learned patterns
- **Trust assessment framework** — Systematic evaluation criteria
- **Bias analysis** — Performance across demographic subgroups

## Key Findings

### Tabular Model
- **Serum creatinine** and **ejection fraction** are the strongest mortality predictors
- Findings align with established cardiology literature
- SHAP and LIME agree on top features, increasing explanation confidence
- Model performance is consistent across age groups (small sample caveat)

### CNN Model
- CNN learns to focus on **biologically diagnostic** flower parts, not background
- Sunflower center disks, rose petal arrangements, tulip cup shapes are correctly identified
- Misclassifications occur between visually similar species (daisy vs. dandelion)
- Grad-CAM provides meaningful but coarse localization

## Report

See `REPORT.md` for the full project report with:
- Introduction and motivation
- Detailed methodology
- Results and metrics
- XAI analysis and insights
- Conclusions and recommendations

## License

This project is for educational purposes as part of an Explainable AI assignment.
