from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained Random Forest model
model = joblib.load("random_forest.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from the form
    data = {
        "loan_amnt": request.form.get("loan_amnt"),
        "funded_amnt": request.form.get("funded_amnt"),
        "funded_amnt_inv": request.form.get("funded_amnt_inv"),
        "term": request.form.get("term"),
        "int_rate": request.form.get("int_rate"),
        "installment": request.form.get("installment"),
        "grade": request.form.get("grade"),
        "sub_grade": request.form.get("sub_grade"),
        "emp_length": request.form.get("emp_length"),
        "home_ownership": request.form.get("home_ownership"),
        "annual_inc": request.form.get("annual_inc"),
        "verification_status": request.form.get("verification_status"),
        "purpose": request.form.get("purpose"),
        "dti": request.form.get("dti"),
        "delinq_2yrs": request.form.get("delinq_2yrs"),
        "inq_last_6mths": request.form.get("inq_last_6mths"),
        "mths_since_last_delinq": request.form.get("mths_since_last_delinq"),
        "mths_since_last_record": request.form.get("mths_since_last_record"),
        "open_acc": request.form.get("open_acc"),
        "pub_rec": request.form.get("pub_rec"),
        "revol_bal": request.form.get("revol_bal"),
        "revol_util": request.form.get("revol_util"),
        "total_acc": request.form.get("total_acc"),
        "pub_rec_bankruptcies": request.form.get("pub_rec_bankruptcies")
    }

    # Convert form data into DataFrame
    df = pd.DataFrame([data])

    # Convert numerical columns
    numeric_columns = [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "inq_last_6mths",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "pub_rec_bankruptcies"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Make prediction
    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "High Default Risk"
    else:
        result = "Low Default Risk"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)
