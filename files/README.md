# Medical Diagnosis with Machine Learning

This project uses machine learning to predict diseases (such as diabetes or heart disease) from patient data. It demonstrates a complete ML pipeline: data loading, preprocessing, EDA, model building, evaluation, and deployment.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing & cleaning
- Multiple ML models (Logistic Regression, Random Forest, etc.)
- Model evaluation & comparison
- Optional Streamlit web app for predictions

## Datasets

- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

- Run the Jupyter notebook for EDA and training:
  ```bash
  jupyter notebook notebooks/eda_and_modeling.ipynb
  ```
- (Optional) Launch the Streamlit app:
  ```bash
  streamlit run app/streamlit_app.py
  ```

## Folder Structure

```
├── app/                   # Streamlit app (optional)
├── data/                  # Raw datasets (not included in repo)
├── models/                # Saved models
├── notebooks/
│   └── eda_and_modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── modeling.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

**Note:** Download datasets into the `data/` folder before running the code.