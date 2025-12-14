# Credit-Risk-Probability-Model-for-Alternative-Data-Week-4
KIAM8 Week 4 Challenge Repo

# Credit Risk Model - Buy-Now-Pay-Later Service

## Project Overview

This project implements an end-to-end credit scoring model for **Bati Bank**, a leading financial service provider partnering with an eCommerce company to enable a buy-now-pay-later service. The model predicts credit risk for potential borrowers using transactional data and machine learning techniques.

### Business Context

Credit scoring assigns a quantitative measure to potential borrowers to estimate the likelihood of default. This project transforms behavioral transaction data into predictive risk signals by analyzing customer Recency, Frequency, and Monetary (RFM) patterns.

**Key Objectives:**
1. Define a proxy variable to categorize users as high risk (bad) or low risk (good)
2. Select observable features that predict default behavior
3. Develop a model that assigns risk probability for new customers
4. Create a credit scoring system from risk probability estimates
5. Predict optimal loan amounts and durations

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability

The Basel II Accord emphasizes risk measurement and requires financial institutions to maintain adequate capital reserves based on credit risk exposure. This regulatory framework influences our modeling approach in several ways:

- **Transparency Requirements:** Regulators and stakeholders need to understand how credit decisions are made, making model interpretability crucial
- **Audit Trail:** An interpretable model with clear documentation enables regulatory compliance and internal audits
- **Risk Quantification:** Basel II requires accurate estimation of Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)
- **Validation Standards:** Models must be validated against independent data and stress-tested under various scenarios

### Proxy Variable Necessity and Business Risks

Since we lack a direct "default" label in the eCommerce transaction data, creating a proxy variable is essential but introduces specific risks:

**Why Proxy Variables are Necessary:**
- The dataset contains transactional behavior but no historical loan performance data
- We must infer creditworthiness from observable patterns like customer engagement and spending behavior
- RFM analysis provides a reasonable proxy by identifying disengaged customers who exhibit risk-like characteristics

**Potential Business Risks:**
- **Proxy Validity:** Customer disengagement doesn't perfectly correlate with loan default - some disengaged customers might still be creditworthy
- **False Negatives:** We may reject creditworthy applicants who appear risky based on limited transaction history
- **False Positives:** We may approve high-risk customers if their transaction patterns don't reveal underlying financial instability
- **Regulatory Scrutiny:** Using proxy variables requires clear documentation and justification to regulators
- **Model Drift:** The relationship between our proxy and true default risk may change over time as customer behavior evolves

### Model Complexity Trade-offs

The choice between simple interpretable models and complex high-performance models involves critical trade-offs in a regulated financial context:

**Simple Models (e.g., Logistic Regression with WoE):**

*Advantages:*
- High interpretability - each feature's contribution is transparent
- Easier to explain to regulators, auditors, and business stakeholders
- Weight of Evidence (WoE) transformation provides clear monotonic relationships
- Faster to deploy and maintain
- Lower computational requirements
- Easier to detect and debug issues

*Disadvantages:*
- May miss complex non-linear patterns in data
- Potentially lower predictive accuracy
- Limited ability to capture feature interactions

**Complex Models (e.g., Gradient Boosting):**

*Advantages:*
- Higher predictive accuracy and discrimination power
- Captures complex patterns and feature interactions
- Better handles non-linear relationships
- Can automatically detect important features

*Disadvantages:*
- "Black box" nature makes regulatory approval challenging
- Difficult to explain individual predictions to customers
- Risk of overfitting, especially with limited data
- Higher computational costs and maintenance overhead
- May not generalize well to new market conditions
- Harder to debug when performance degrades

**Recommended Approach:**
In practice, a hybrid strategy often works best - develop both model types, use the complex model for performance benchmarking, but deploy the simpler model for production with appropriate safeguards. Consider ensemble methods or explainable AI techniques (SHAP values, LIME) if deploying complex models.

## Dataset

**Source:** [Xente Challenge | Kaggle](https://www.kaggle.com/datasets/atwine/xente-challenge)

### Data Fields

| Field | Description |
|-------|-------------|
| `TransactionId` | Unique transaction identifier |
| `BatchId` | Unique number for batch processing |
| `AccountId` | Unique customer account identifier |
| `SubscriptionId` | Unique customer subscription identifier |
| `CustomerId` | Unique identifier attached to account |
| `CurrencyCode` | Country currency |
| `CountryCode` | Numerical geographical code |
| `ProviderId` | Source provider of purchased item |
| `ProductId` | Item name being bought |
| `ProductCategory` | Broader product categories |
| `ChannelId` | Platform used (web, Android, iOS, pay later, checkout) |
| `Amount` | Transaction value (positive for debits, negative for credits) |
| `Value` | Absolute value of amount |
| `TransactionStartTime` | Transaction start time |
| `PricingStrategy` | Xente's pricing structure category |
| `FraudResult` | Fraud status (1 = yes, 0 = no) |

## Project Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline configuration
├── data/                        # (add to .gitignore)
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data for training
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering scripts
│   ├── train.py                 # Model training scripts
│   ├── predict.py               # Inference scripts
│   └── api/
│       ├── main.py              # FastAPI application
│       └── pydantic_models.py   # API data models
├── tests/
│   └── test_data_processing.py  # Unit tests
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/GrimVad3r/Credit-Risk-Probability-Model-for-Alternative-Data-Week-4
cd credit-risk-model
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
   - Place raw data files in `data/raw/`

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/eda.ipynb
```

### 2. Feature Engineering and Model Training
```bash
python src/data_processing.py
python src/train.py
```

### 3. Run MLflow UI
```bash
mlflow ui
```
Access at `http://localhost:5000`

### 4. Start API Server
```bash
uvicorn src.api.main:app --reload
```
API documentation at `http://localhost:8000/docs`

### 5. Docker Deployment
```bash
docker-compose up --build
```

## Model Development Pipeline

### Task 1: Understanding Credit Risk
- Review Basel II Capital Accord requirements
- Understand credit scoring fundamentals
- Document business understanding

### Task 2: Exploratory Data Analysis
- Data structure and quality assessment
- Summary statistics and distributions
- Correlation analysis
- Missing value and outlier detection

### Task 3: Feature Engineering
- Aggregate features (total, average, count, std dev)
- Time-based features (hour, day, month, year)
- Categorical encoding (one-hot, label encoding)
- Missing value handling
- Normalization/standardization
- Weight of Evidence (WoE) and Information Value (IV) transformations

### Task 4: Proxy Target Variable Engineering
- Calculate RFM (Recency, Frequency, Monetary) metrics
- K-Means clustering (3 clusters)
- Define high-risk customer segment
- Create binary `is_high_risk` target variable

### Task 5: Model Training and Tracking
- Train multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Hyperparameter tuning (Grid Search, Random Search)
- Experiment tracking with MLflow
- Model evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Unit testing

### Task 6: Model Deployment and CI/CD
- FastAPI REST API with `/predict` endpoint
- Pydantic models for data validation
- Docker containerization
- GitHub Actions CI/CD pipeline
- Automated linting and testing

## API Usage

### Prediction Endpoint

**POST** `/predict`

**Request Body:**
```json
{
  "customer_id": "12345",
  "recency": 30,
  "frequency": 15,
  "monetary": 5000,
  "transaction_count": 20,
  "avg_transaction_amount": 250
}
```

**Response:**
```json
{
  "customer_id": "12345",
  "risk_probability": 0.23,
  "risk_category": "low",
  "credit_score": 720
}
```

## Testing

Run unit tests:
```bash
pytest tests/
```

Run linter:
```bash
flake8 src/
```
## Credit Scoring Business 



## 📄 Financial Risk Modeling: Basel II, Proxy Variables, and Model Selection

This below articles provide a concise overview of key considerations in financial risk modeling, focusing on the influence of regulatory mandates (Basel II), the necessity of proxy variables, and the strategic trade-offs in model selection.

---

### 1. 🔍 Basel II Accord and the Imperative for Model Interpretability

The **Basel II Accord**, specifically the Internal Ratings Based (IRB) approach, mandates that financial institutions use internal models to estimate crucial risk parameters, such as the Probability of Default (PD), for calculating regulatory capital.

* **Regulatory Requirement:** Basel II fundamentally shifts risk measurement responsibility to the bank.
* **Need for Transparency:** To secure **supervisory approval** for these models, they must be highly **interpretable** and **well-documented**.
* **Auditability:** Regulators, auditors, and internal risk managers must be able to:
    * Understand the precise logical flow of the model.
    * Validate the relationship between inputs and the risk estimate.
    * Critically assess and explain the model's output, ensuring it is a reliable tool for capital adequacy and decision-making, not a "black box." 

---

### 2. 🎯 Proxy Variables: Necessity and Associated Risks

In credit risk modeling, the true, legal event of "default" is often too rare (especially in high-quality portfolios) to provide sufficient data for robust statistical model training.

#### 💡 Necessity of a Proxy

A **proxy variable** (e.g., being **90 days past due (DPD)** on any payment) becomes essential to:

1.  **Increase the Sample Size:** Provide a statistically sufficient number of 'bad' outcomes for reliable model fitting.
2.  **Capture Early Warning Signals:** Model credit deterioration earlier than the true, legal default event.

#### ⚠️ Potential Business Risks

Using an imperfect proxy introduces significant risk:

* **Model Misspecification:** The model is optimized to predict the proxy, not the true, underlying legal default. This discrepancy can lead to the model making **inaccurate predictions** of actual losses.
* **Inaccurate Capital Calculation:** If the model's PD estimates are flawed, the bank's **Risk-Weighted Assets (RWA)** calculation will be incorrect, potentially resulting in miscalculated regulatory capital.
* **Suboptimal Decisions:** Inaccurate loss rate predictions can cause the bank to:
    * **Overprice** credit (losing profitable customers).
    * **Underprice** credit (taking on excessive, uncompensated risk), impacting long-term profitability and stability.

---

### 3. ⚖️ Model Selection Trade-Offs in a Regulated Environment

The choice between a simple, transparent model (like Logistic Regression) and a complex, high-performance model (like Gradient Boosting) involves a critical trade-off between **Interpretability** and **Predictive Accuracy**.

| Feature | Simple Model (e.g., Logistic Regression with WoE) | Complex Model (e.g., Gradient Boosting/Deep Learning) |
| :--- | :--- | :--- |
| **Primary Advantage** | **High Interpretability and Explainability.** | **High Predictive Performance/Accuracy.** |
| **Regulatory Standing** | **Generally Preferred.** Easily meets requirements for justification, auditability, and clear documentation. | **Challenging.** Requires extensive effort and sophisticated post-hoc tools (like SHAP/LIME) to justify decisions to regulators. |
| **Model Structure** | Linear, coefficients clearly show the direction and magnitude of risk factors. | Non-linear, captures complex interactions and non-monotonic relationships between variables. |
| **Monitoring Overhead** | **Low.** Simple, stable coefficients, easy to monitor for drift (e.g., PSI). | **High.** Difficult to interpret changes; requires more robust monitoring for drift and decay. |

In regulated finance, **transparency and auditability** often supersede marginal gains in predictive performance, making simpler, interpretable models the standard for regulatory capital and credit decision models.

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs code linting (flake8)
2. Executes unit tests (pytest)
3. Fails the build if either step fails

Triggered on every push to the main branch.

## Key Technologies

- **ML Framework:** scikit-learn, XGBoost, LightGBM
- **Experiment Tracking:** MLflow
- **API Framework:** FastAPI
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Feature Engineering:** xverse, woe
- **Testing:** pytest
- **Containerization:** Docker
- **CI/CD:** GitHub Actions

## References

### Credit Risk Fundamentals
- [Investopedia - Credit Risk](https://www.investopedia.com/terms/c/creditrisk.asp)
- [Corporate Finance Institute - Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Credit Risk Officer Guide](https://www.risk-officer.com/Credit_Risk.htm)

### Feature Engineering
- [Weight of Evidence and Information Value](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
- [xverse Python Package](https://pypi.org/project/xverse/)
- [woe Python Package](https://pypi.org/project/woe/)

### Machine Learning & MLOps
- [Hyperparameter Tuning with GridSearchCV](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/)
- [Random Search vs Grid Search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
- [Sklearn Pipelines Guide](https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf)