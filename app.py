import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "loan_model.pkl"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if not os.path.exists(MODEL_PATH):
        return render_template(
            "index.html",
            error="Model file not found. Please run python train.py first."
        )

    try:
        # Load the trained machine learning model
        model = joblib.load(MODEL_PATH)

        # Collect input values from the HTML form
        input_data = {
            "Gender": request.form.get("gender", "Male"),
            "Married": request.form.get("married", "Yes"),
            "Education": request.form.get("education", "Graduate"),
            "Self_Employed": request.form.get("self_employed", "No"),
            "ApplicantIncome": float(
                request.form.get("applicant_income", 0)
            ),
            "CoapplicantIncome": float(
                request.form.get("coapplicant_income", 0)
            ),
            "LoanAmount": float(
                request.form.get("loan_amount", 0)
            ),
            "Loan_Amount_Term": float(
                request.form.get("loan_term", 360)
            ),
            "Credit_History": float(
                request.form.get("credit_history", 1)
            ),
            "Property_Area": request.form.get(
                "property_area", "Urban"
            )
        }

        # Convert the applicant details into a DataFrame
        query_df = pd.DataFrame([input_data])

        # Predict loan status
        prediction = model.predict(query_df)[0]

        # Obtain the probability corresponding to the predicted class
        probabilities = model.predict_proba(query_df)[0]
        classes = list(model.classes_)
        confidence = float(
            probabilities[classes.index(prediction)] * 100
        )

        # Handle either Y/N labels or 1/0 labels
        approved = str(prediction).strip().upper() in (
            "Y", "1", "1.0", "TRUE", "APPROVED"
        )

        if approved:
            result_text = "Loan Approved"
            result_class = "approved"
            reasons = [
                "The model predicts that the application meets "
                "the approval patterns learned from the training data."
            ]
        else:
            result_text = "Loan Rejected"
            result_class = "rejected"
            reasons = []

            # Explain relevant input factors without claiming
            # that they are the model's exact decision logic.
            if input_data["Credit_History"] == 0:
                reasons.append(
                    "Credit history is marked as not satisfactory."
                )

            if input_data["ApplicantIncome"] <= 0:
                reasons.append(
                    "Applicant income must be greater than zero."
                )

            if (
                input_data["LoanAmount"] > 0
                and input_data["ApplicantIncome"]
                + input_data["CoapplicantIncome"] > 0
                and input_data["LoanAmount"]
                > (
                    input_data["ApplicantIncome"]
                    + input_data["CoapplicantIncome"]
                ) / 10
            ):
                reasons.append(
                    "The requested loan amount is relatively high "
                    "compared with the combined monthly income."
                )

            if not reasons:
                reasons.append(
                    "The model predicted rejection based on patterns "
                    "in the training dataset. This is not an exact "
                    "explanation of the model's decision."
                )

        return render_template(
            "index.html",
            prediction_text=result_text,
            result_class=result_class,
            confidence_score=f"{confidence:.1f}%",
            reasons=reasons
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values for income, "
                  "loan amount, loan term, and credit history."
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5004)
