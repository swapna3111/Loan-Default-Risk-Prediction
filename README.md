
# Loan Default Risk Prediction Using Machine Learning

## Project Overview

This project is a machine learning-based application that analyzes loan applicant information and predicts a loan status category based on historical loan data.

The application uses machine learning algorithms and a Flask web interface to accept applicant details and display a prediction.

## Objectives

- Analyze loan-related data using machine learning.
- Clean and preprocess the dataset.
- Train and compare multiple classification models.
- Build a simple web application for predictions.
- Understand how machine learning can support loan risk analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- HTML and CSS
- Joblib

## Machine Learning Models

The following models were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The Random Forest model was selected for the web application based on the evaluation results.

## Project Workflow

1. Load the loan dataset.
2. Clean the data and remove unnecessary columns.
3. Handle missing values and preprocess numerical and categorical features.
4. Split the data into training and testing sets.
5. Train and evaluate the machine learning models.
6. Save the trained model using Joblib.
7. Build a Flask web application.
8. Enter applicant details and view the model's prediction.

## Project Structure

```text
Loan-Default-Risk-Prediction/
├── app.py
├── train_model.py
├── compare_models.py
├── prepare_data.py
├── data_check.py
├── test_ml.py
├── columns.txt
├── requirements.txt
├── random_forest.pkl
├── decision_tree.pkl
├── logistic_regression.pkl
├── loan_default_model.pkl
├── model_results.csv
├── static/
│   └── style.css
└── templates/
    └── index.html
```

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

The Random Forest model achieved approximately 67.80% accuracy and an F1-score of 35.66% on the test set.

These results are from a single train-test split and do not establish real-world lending performance.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/swapna3111/Loan-Default-Risk-Prediction.git
```

### 2. Open the project folder

```bash
cd Loan-Default-Risk-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment on Windows

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open the application

Visit:

http://127.0.0.1:5000

## Future Enhancements

- Improve model performance through further evaluation and tuning.
- Add a prediction probability display.
- Improve the user interface.
- Add input validation and clearer prediction explanations.
- Deploy the application online.

## Disclaimer

This project is an educational machine learning prototype. It is not intended to make real loan approval or rejection decisions. Predictions depend on the dataset, model, and verified meaning of the target labels.
