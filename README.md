# 🧠 Responsible AI Bias Detection System

A web-based application that detects and analyzes bias in machine learning models using fairness metrics, explainable AI techniques, and AI-generated insights.

---

## 🚀 Overview

This project helps identify **bias in AI/ML models** by evaluating predictions across sensitive attributes such as gender or other categorical features. It provides both **quantitative metrics** and **qualitative insights** to support responsible AI development.

---

## ✨ Key Features

* 📊 **Fairness Metrics**

  * Demographic Parity (DP)
  * Equal Opportunity (EO)
  * Predictive Equality (PE)

* 🧮 **Automated Bias Scoring**

  * Calculates average bias percentage
  * Classifies model as:

    * Fair
    * Slight Bias
    * Moderate Bias
    * Severe Bias

* 🤖 **AI-Powered Insights**

  * Uses Gemini API to generate human-readable bias explanations

* 📈 **Visualization**

  * Metric comparison charts
  * SHAP-based explainability

* 📄 **PDF Report Generation**

  * Downloadable report with metrics + AI insights

* 📂 **Flexible Input**

  * Upload:

    * Dataset (.csv / .xlsx)
    * Trained ML model (.pkl)

---

## 🏗️ System Architecture

1. User uploads dataset and model
2. Model generates predictions
3. Sensitive attribute is detected automatically
4. Fairness metrics are computed
5. Bias level is classified
6. AI generates summary insights
7. Charts and PDF report are generated

---

## 🛠️ Tech Stack

* **Backend:** Flask (Python)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Fairness Metrics:** Fairlearn
* **Visualization:** Matplotlib
* **Explainability:** SHAP
* **Report Generation:** ReportLab
* **AI Integration:** Gemini API

---

## 📁 Project Structure

```
├── app.py
├── gemini_summary.py
├── requirements.txt
├── Procfile
├── templates/
│   └── index.html
├── static/
├── uploads/
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set environment variable

Create a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the application

```
python app.py
```

---

## 🌐 Deployment

This project can be deployed using platforms like:

* Render
* Railway

Make sure to:

* Add `GEMINI_API_KEY` in environment variables
* Use `gunicorn` for production

---

## 📊 Example Use Case

Upload a loan approval dataset and trained model to:

* Detect bias based on gender
* Evaluate fairness metrics
* Generate AI-based explanation
* Download a complete bias report

---

## ⚠️ Limitations

* Large datasets may increase processing time
* SHAP may require high computational resources (optional)
* Cloud deployment may restrict heavy computations

---

## 🔮 Future Enhancements

* Real-time bias monitoring
* Dashboard analytics
* Support for deep learning models
* Advanced explainability techniques (LIME, PDP)
* User authentication system

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork this repo and improve the system.

---

## 📜 License

This project is for educational and research purposes.

---

## 👨‍💻 Author

**Abhimanyu M B**
AI/ML Student | Responsible AI Enthusiast

---

## 🌟 Final Note

This project demonstrates the importance of **fairness, transparency, and accountability in AI systems**, aligning with modern Responsible AI principles.
