# 🛒 Retail Customer Purchase Prediction

Predicts whether an e-commerce website visitor will **purchase (Yes/No)** and gives a
**purchase probability (%)**, based on their real-time browsing behavior — pages viewed,
time on site, previous visits, cart activity, device type, and purchase history.

**ML Type:** Binary Classification
**Models:** Logistic Regression vs. Random Forest (best one auto-selected by ROC-AUC)
**Demo:** Live interactive Streamlit web app

---

## 📁 Project Structure

```
retail-purchase-prediction/
├── data/
│   ├── generate_data.py      # creates synthetic customer behavior dataset
│   └── customer_data.csv     # generated dataset (5000 rows)
├── model/
│   ├── purchase_model.pkl    # trained pipeline (preprocessing + model)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_comparison.csv
├── train_model.py            # trains, evaluates, and saves the best model
├── app.py                    # Streamlit demo app (run this live at the hackathon)
├── requirements.txt
└── README.md
```

---

## 🚀 Step-by-Step Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_data.py
```
This creates `data/customer_data.csv` with 5,000 simulated customer sessions
(pages viewed, time spent, cart activity, device type, purchase history, and
the actual purchase outcome).

> **Using real data instead?** Replace this step with your own CSV as long as it
> has the same column names (see Features section below).

### 3. Train the model
```bash
python train_model.py
```
This will:
- Split data into train/test sets (80/20, stratified)
- Train **Logistic Regression** and **Random Forest**
- Print accuracy, precision, recall, F1, and ROC-AUC for both
- Auto-select the best model by ROC-AUC
- Save the trained pipeline to `model/purchase_model.pkl`
- Save confusion matrix, ROC curve, and feature importance charts

Expected output (your numbers may vary slightly):
```
                    accuracy  precision  recall      f1  roc_auc
LogisticRegression     0.831     0.7455  0.6797  0.7111   0.8886
RandomForest           0.800     0.6410  0.7876  0.7067   0.8813
```

### 4. Launch the live demo app
```bash
streamlit run app.py
```
Opens a browser at `http://localhost:8501` with:
- **Tab 1 — Live Prediction:** sliders to enter a customer's behavior and get an
  instant Purchase Yes/No + probability %, with quick presets for "Customer A" (high-intent)
  and "Customer B" (low-intent) exactly like your example.
- **Tab 2 — Model Performance:** comparison table, confusion matrix, ROC curve, and
  feature importance chart — great for judges who want to see the ML rigor.

---

## 🧠 Features Used

| Feature | Type | Description |
|---|---|---|
| `pages_viewed` | numeric | Number of pages visited in session |
| `time_spent_sec` | numeric | Time spent on site (seconds) |
| `previous_visits` | numeric | Number of prior visits |
| `products_viewed` | numeric | Number of distinct products viewed |
| `added_to_cart` | binary | 1 if any product added to cart, else 0 |
| `device_type` | categorical | mobile / desktop / tablet |
| `previous_purchases` | numeric | Number of past purchases by this customer |

**Target:** `purchase` (1 = Yes, 0 = No)

---

## 📊 Why Logistic Regression *and* Random Forest?

Comparing two models — a simple, interpretable linear model and a more flexible
ensemble model — is a strong hackathon talking point:
- **Logistic Regression** gives clean, explainable coefficients ("cart activity adds
  X to purchase odds") — good for business stakeholders.
- **Random Forest** captures non-linear interactions (e.g., "high pages + no cart"
  behaving differently than expected) and gives feature importance rankings.

The script automatically picks the better one by ROC-AUC, but printing both
comparisons is exactly the kind of rigor judges look for.


## 🔧 Extending This Project 

- **Deploy the model as a REST API** with FastAPI/Flask (`predict` endpoint) so it
  could plug into a real website.
- **Add SHAP explainability** to show *why* a specific customer got their score.
- **Add a `time_of_day` / `day_of_week` feature** — purchase behavior often varies by time.
- **Try XGBoost/LightGBM** for a performance boost and compare against the two
  baseline models.
- **Connect to a real dataset** — the UCI "Online Shoppers Purchasing Intention"
  dataset has nearly identical features and is a drop-in replacement for
  `customer_data.csv` if you want real-world data instead of synthetic.

---

## ⚠️ Note on the Data

This project uses a **synthetic dataset** generated with a known underlying logic
(cart activity, previous purchases, and engagement time are the strongest purchase
signals) so the model has clean patterns to learn — ideal for a hackathon demo where
reliability during a live presentation matters more than real-world noise. Swap in
real data any time using the same column schema.
