# stance-detection-thesis-experiments

This repository contains all code for:

- **Baselines** (majority & uniform)
- **Logistic Regression** (majority‑vote & annotator‑wise)
- **BERT fine‑tuning** (majority‑vote & annotator‑wise)
- **Evaluation scripts** (hard/soft metrics, disagreement analysis)

1. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # on Windows: venv\Scripts\activate

   pip install -r requirements.txt

