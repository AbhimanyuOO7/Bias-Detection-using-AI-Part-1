from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from gemini_summary import generate_bias_summary

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# ================= DETECT SENSITIVE COLUMN =================
def detect_sensitive_column(df):
    priority = ["Gender", "gender", "Sex", "sex"]

    for col in priority:
        if col in df.columns:
            return col

    for col in df.columns:
        if df[col].dtype == object:
            return col

    return df.columns[0]


# ================= PREDICTIVE EQUALITY =================
def predictive_equality_difference(y_true, y_pred, sensitive):
    groups = np.unique(sensitive)
    fprs = []

    for g in groups:
        idx = sensitive == g

        cm = confusion_matrix(
            y_true[idx],
            y_pred[idx],
            labels=[0, 1]
        )

        TN, FP, FN, TP = cm.ravel()

        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
        fprs.append(fpr)

    return abs(max(fprs) - min(fprs))


# ================= SHAP FUNCTION =================
def generate_shap_plot(model, X):
    try:
        X_sample = X.sample(min(100, len(X)))

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)

        plot_path = os.path.join(STATIC_FOLDER, "shap_plot.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

        return "shap_plot.png"

    except Exception as e:
        print("SHAP Error:", e)
        return None


# ================= METRIC CHARTS =================
def generate_metric_charts(result):
    metrics = {
        "dp": result["dp"],
        "eo": result["eo"],
        "pe": result["pe"]
    }

    filenames = {}

    for key, value in metrics.items():
        plt.figure()
        plt.bar([key.upper()], [value])
        plt.ylim(0, 1)
        plt.title(f"{key.upper()} Value")

        path = os.path.join(STATIC_FOLDER, f"{key}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        filenames[key] = f"{key}.png"

    return filenames


# ================= PDF GENERATION =================
def generate_pdf(result):
    pdf_path = os.path.join(STATIC_FOLDER, "report.pdf")

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Bias Detection Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Bias Level: {result['bias_level']}", styles['Normal']))
    elements.append(Paragraph(f"Demographic Parity: {result['dp']}", styles['Normal']))
    elements.append(Paragraph(f"Equal Opportunity: {result['eo']}", styles['Normal']))
    elements.append(Paragraph(f"Predictive Equality: {result['pe']}", styles['Normal']))
    elements.append(Paragraph(f"Average Bias: {result['avg']}%", styles['Normal']))

    elements.append(Spacer(1, 10))

    elements.append(Paragraph("AI Insight:", styles['Heading2']))
    elements.append(Paragraph(result["summary"], styles['Normal']))

    elements.append(Spacer(1, 10))

    # SHAP image
    try:
        shap_path = os.path.join(STATIC_FOLDER, "shap_plot.png")
        if os.path.exists(shap_path):
            elements.append(Image(shap_path, width=400, height=300))
    except:
        pass

    doc.build(elements)

    return "report.pdf"


# ================= MAIN BIAS FUNCTION =================
def calculate_bias(model, df):
    target_col = df.columns[-1]
    sensitive_col = detect_sensitive_column(df)

    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    sensitive = df[sensitive_col]

    # MODEL PREDICTION
    try:
        y_pred = model.predict(X)
    except:
        X_encoded = pd.get_dummies(X)

        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            for col in expected:
                if col not in X_encoded.columns:
                    X_encoded[col] = 0
            X_encoded = X_encoded[expected]

        y_pred = model.predict(X_encoded)

    # ENCODING
    y_true_encoded = pd.factorize(y_true)[0].astype(int)
    sensitive_encoded = pd.factorize(sensitive)[0].astype(int)
    y_pred_encoded = pd.factorize(y_pred)[0].astype(int)

    # FAIRNESS METRICS
    dp = demographic_parity_difference(
        y_true=y_true_encoded,
        y_pred=y_pred_encoded,
        sensitive_features=sensitive_encoded
    )

    eo = equalized_odds_difference(
        y_true=y_true_encoded,
        y_pred=y_pred_encoded,
        sensitive_features=sensitive_encoded
    )

    pe = predictive_equality_difference(
        y_true_encoded,
        y_pred_encoded,
        sensitive_encoded
    )

    avg_bias = (abs(dp) + abs(eo) + abs(pe)) / 3
    avg_bias_percent = avg_bias * 100

    if avg_bias_percent <= 5:
        bias_level = "Fair"
    elif avg_bias_percent <= 10:
        bias_level = "Slight Bias"
    elif avg_bias_percent <= 20:
        bias_level = "Moderate Bias"
    else:
        bias_level = "Severe Bias"

    result = {
        "bias_level": bias_level,
        "dp": round(abs(dp), 3),
        "eo": round(abs(eo), 3),
        "pe": round(abs(pe), 3),
        "avg": round(avg_bias_percent, 2)
    }

    result["charts"] = generate_metric_charts(result)
    result["shap_plot"] = generate_shap_plot(model, X)

    try:
        result["summary"] = generate_bias_summary(
            result["dp"],
            result["eo"],
            result["pe"],
            result["avg"],
            result["bias_level"]
        )
    except Exception as e:
        print("Gemini Error:", e)
        result["summary"] = "AI summary generation failed."

    result["pdf"] = generate_pdf(result)

    return result


# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            dataset_file = request.files["dataset"]
            model_file = request.files["model"]

            dataset_path = os.path.join(UPLOAD_FOLDER, dataset_file.filename)
            model_path = os.path.join(UPLOAD_FOLDER, model_file.filename)

            dataset_file.save(dataset_path)
            model_file.save(model_path)

            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_excel(dataset_path)

            df = df.dropna()

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            result = calculate_bias(model, df)

        except Exception as e:
            print("ERROR:", e)
            result = {"error": str(e)}

    return render_template("index.html", result=result)


# ================= RUN =================
if __name__ == "__main__":
    print("Responsible AI Bias Detection System Running...")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
