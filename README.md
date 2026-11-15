*# Industrial AI Predictive Maintenance*



*An AI-powered predictive maintenance dashboard built using Streamlit and machine learning. The application analyzes industrial equipment data, predicts potential machine failures, detects anomalies, and presents insights through interactive visualizations.*



*## Features*



*- \*\*Machine Failure Prediction:\*\* Predicts the possibility of equipment failure using a trained XGBoost model.*

*- \*\*Anomaly Detection:\*\* Identifies unusual equipment behavior using Isolation Forest.*

*- \*\*Model Explainability:\*\* Uses SHAP and feature importance to help interpret model predictions.*

*- \*\*Interactive Dashboard:\*\* Presents analytical insights and visualizations using Plotly.*

*- \*\*Model Comparison:\*\* Displays the performance comparison of trained machine learning models.*

*- \*\*Data Exploration:\*\* Analyzes the AI4I 2020 Predictive Maintenance dataset.*



*## Technology Stack*



*| Category | Technologies |*

*|---|---|*

*| Programming Language | Python |*

*| Dashboard | Streamlit |*

*| Data Processing | Pandas, NumPy |*

*| Machine Learning | Scikit-learn, XGBoost |*

*| Explainable AI | SHAP |*

*| Visualization | Plotly, Matplotlib, Seaborn |*

*| Model Persistence | Joblib |*



*## Project Structure*



*```text*

*Industrial-Predictive-Maintenance/*

*│*

*├── app.py*

*├── requirements.txt*

*├── .gitignore*

*│*

*├── dataset/*

*│   └── ai4i2020.csv*

*│*

*├── models/*

*│   ├── best\_model.pkl*

*│   ├── decision\_tree.pkl*

*│   ├── isolation\_forest.pkl*

*│   ├── logistic\_regression.pkl*

*│   ├── random\_forest.pkl*

*│   └── xgboost.pkl*

*│*

*├── results/*

*│   ├── feature\_importance.csv*

*│   ├── feature\_importance.png*

*│   └── model\_comparison.csv*

*│*

*└── src/*

&#x20;   *├── anomaly\_detection.py*

&#x20;   *├── data\_exploration.py*

&#x20;   *├── explainability.py*

&#x20;   *├── prediction.py*

&#x20;   *├── preprocessing.py*

&#x20;   *└── train\_models.py*

*```*



*## Run Locally*



*### 1. Clone the repository*



*```bash*

*git clone YOUR\_GITHUB\_REPOSITORY\_URL*

*cd Industrial-Predictive-Maintenance*

*```*



*### 2. Create and activate a virtual environment*



*```bash*

*python -m venv venv*

*```*



*Windows PowerShell:*



*```powershell*

*.\\venv\\Scripts\\Activate.ps1*

*```*



*### 3. Install dependencies*



*```bash*

*pip install -r requirements.txt*

*```*



*### 4. Launch the dashboard*



*```bash*

*streamlit run app.py*

*```*



*The application will be available at:*



*```text*

*http://localhost:8501*

*```*



*## Dataset*



*This project uses the AI4I 2020 Predictive Maintenance Dataset, which contains simulated industrial process and machine data for predictive maintenance research.*



*Dataset source: \[UCI Machine Learning Repository – AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)*



*## Project Objective*



*To demonstrate how machine learning, anomaly detection, and explainable AI can be combined in an interactive predictive maintenance application to support industrial equipment monitoring and maintenance analysis.*

