# 🏠 California Housing — ML Pipeline, API and Frontend

Complete **Machine Learning** project to predict the median house value (`median_house_value`) in California districts. Includes **exploratory data analysis (EDA)**, **comparative model training**, **REST API** with **FastAPI** and **frontend** in **Streamlit** for individual and batch predictions (CSV).

---

## 📌 Main Features

- **EDA**: descriptive analysis, variable distribution, null detection, correlations.
- **Preprocessing**: value imputation, scaling of numerical features and one-hot encoding for categorical features.
- **Models tested**:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `XGBRegressor`
- Automatically saves the **best model by RMSE** in `artifacts/model.joblib`.
- **REST API**: `/predict` endpoint to receive data and return estimated price.
- **Frontend**:
  - **Individual** prediction via form (API).
  - **Batch** prediction from a CSV file processed locally.
- **Tests** for API and pipeline with `pytest`.

---

## 📂 Project Structure

```
california-housing-ml/
├── .streamlit/
│   └── secrets.toml          # Optional config (API_URL for Streamlit)
├── api/
│   └── app.py               # FastAPI API with /predict and /health
├── app/
│   └── streamlit_app.py     # Streamlit App (individual + CSV)
├── artifacts/
│   └── model.joblib         # Trained pipeline (preprocessing + model)
├── data/
│   └── housing.csv          # Original dataset (optional)
├── eda/
│   └── eda_california_housing.ipynb  # Exploratory analysis
├── src/
│   ├── __init__.py
│   ├── config.py            # General configuration
│   ├── data.py              # Data loading functions
│   ├── evaluate.py          # Model evaluation
│   ├── pipeline.py          # Transformation definitions
│   ├── predict.py           # Local prediction
│   ├── train.py             # Model training
│   └── utils.py             # Helper functions
├── tests/
│   ├── test_api.py          # API tests
│   └── test_pipeline.py     # Pipeline tests
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your_username/california-housing-ml.git
cd california-housing-ml

# 2. Create virtual environment
python -m venv .venv
# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

💡 **If you have issues with xgboost on Windows:**

```bash
conda install -c conda-forge xgboost
```

## 🧪 Training and Evaluation

```bash
# Train and save the best model in artifacts/model.joblib
python -m src.train

# Evaluate the trained model
python -m src.evaluate
```

**Example results (may vary):**

| Model | MAE | RMSE | R² |
|--------|-----|------|-----|
| LinearRegression | 50670.49 | 70059.19 | 0.625 |
| RandomForest | 31393.36 | 48676.22 | 0.819 |
| XGBRegressor | 30235.84 | 45930.94 | 0.839 |

## 🚀 API with FastAPI

```bash
uvicorn api.app:app --reload
```

**Interactive documentation:** http://127.0.0.1:8000/docs

### Endpoints:

- **GET** `/health` → check status
- **POST** `/predict` → predict house price

**Example JSON:**

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

**Response:**

```json
{ "predicted_price": 426046.59 }
```

## 🖥️ Frontend with Streamlit

Launches a form for individual prediction and CSV upload for batch prediction.

```bash
streamlit run app/streamlit_app.py
```

### Optional config for production

`.streamlit/secrets.toml`:

```toml
API_URL = "https://my-api.com/predict"
```

### CSV format for batch prediction:

```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
```

💡 **Remember that** `total_rooms`, `total_bedrooms`, `population` and `households` are aggregated at the district level.

## 📊 EDA

In `eda/eda_california_housing.ipynb` you'll find:

- General information (`df.info()`, `df.describe()`).
- Distribution of numerical and categorical variables.
- Correlation heatmap.
- Geographic relationship between location (latitude / longitude) and price.

## 🧰 Tests

```bash
pytest -q
```

- `test_pipeline.py`: checks that the pipeline trains and predicts correctly.
- `test_api.py`: tests the `/health` and `/predict` endpoints.

## 📦 requirements.txt

```txt
pandas
numpy
scikit-learn
xgboost
fastapi
uvicorn
pydantic
joblib
matplotlib
seaborn
streamlit
requests
ipykernel
pytest
```

## 🧹 Recommended .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.DS_Store

# Environments
.venv/
venv/
.env
.env.*

# Streamlit
.streamlit/secrets.toml

# Data and artifacts
# artifacts/
# data/
```
