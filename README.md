# 🏦 Loan Eligibility Prediction Web App

An end-to-end Machine Learning web application that predicts loan approval status using a **Random Forest Classifier** served through a **Flask** interface.

---

## 📁 Project Structure

```text
loan-approval-prediction/
├── static/
│   └── style.css          # UI styling (brown & cream theme)
├── templates/
│   └── index.html         # Web application UI
├── app.py                 # Flask server & prediction routing
├── train.py               # Model training & pipeline export
├── loan_data.csv          # Training dataset
└── loan_model.pkl         # Serialized ML pipeline artifact

```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install flask pandas scikit-learn joblib

```

### 2. Train the Model

Train the Random Forest pipeline and generate the model artifact:

```bash
python train.py

```

### 3. Start the Web App

Run the local Flask development server:

```bash
python app.py

```

### 4. Open in Browser

Visit **`[http://127.0.0.1:5000](http://127.0.0.1:5000)`** to submit applicant data and view eligibility predictions.

---

## 🛠️ Tech Stack

* **Machine Learning:** `scikit-learn` (Random Forest, ColumnTransformer, Pipelines)
* **Data Processing:** `pandas`, `joblib`
* **Backend:** `Flask` (Python)
* **Frontend:** HTML5, CSS3
