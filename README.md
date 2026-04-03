# 🫀 CardioSense — AWS Heart Disease Risk Prediction System

[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Lambda%20%7C%20DynamoDB%20%7C%20SNS-orange?logo=amazon-aws)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-brightgreen)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-blueviolet)](https://shap.readthedocs.io/)

> An end-to-end, cloud-native heart disease risk prediction pipeline — from raw clinical data to real-time alerts — built on AWS.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training on SageMaker](#2-model-training-on-sagemaker)
  - [3. Endpoint Deployment](#3-endpoint-deployment)
  - [4. Lambda Function](#4-lambda-function)
  - [5. Inference & Alerts](#5-inference--alerts)
  - [6. Explainability with SHAP](#6-explainability-with-shap)
- [AWS Services Used](#-aws-services-used)
- [Model Details](#-model-details)
- [API Reference](#-api-reference)
- [SHAP Visualizations](#-shap-visualizations)
- [Getting Started](#-getting-started)
- [Environment Setup](#-environment-setup)

---

## 🔍 Overview

**CardioSense** is a serverless, cloud-native machine learning system that predicts a patient's risk of heart disease in real time. It ingests 13 clinical features (age, cholesterol, chest pain type, etc.), runs them through a trained XGBoost model hosted on Amazon SageMaker, and instantly:

- Stores the prediction result in **Amazon DynamoDB**
- Sends an **SNS email alert** if the patient is classified as **HIGH RISK**
- Returns a structured JSON response with the risk score and prediction label

The system is fully explainable via **SHAP (SHapley Additive exPlanations)**, providing per-feature importance plots so clinicians can understand *why* a particular prediction was made.

---

## 🏗 Architecture

```
Patient Data (JSON)
        │
        ▼
  API Gateway (HTTP)
        │
        ▼
  AWS Lambda Function
   ├─► Amazon SageMaker Endpoint  ──► Risk Score (0–1)
   ├─► Amazon DynamoDB            ──► Persist prediction record
   └─► Amazon SNS                 ──► Email alert (if HIGH RISK)
        │
        ▼
   JSON Response
   { patient_id, risk_score, prediction, timestamp }
```

---

## 📊 Dataset

- **Source:** [UCI Heart Disease Dataset (Cleveland)](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
- **Records:** 303 samples (297 after cleaning)
- **Target:** Binary classification
  - `0` → No Heart Disease (**LOW RISK**)
  - `1` → Heart Disease Present (**HIGH RISK**)
- **Train / Test Split:** 80% / 20% (stratified)

### Features

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | Gender (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |

---

## 📁 Project Structure

```
Aws Health Monitoring/
│
├── Untitled.ipynb              # Main Jupyter notebook (full pipeline)
├── heart_processed.csv         # Cleaned dataset (all 297 records)
├── train.csv                   # SageMaker training data (no header, label-first)
├── test.csv                    # SageMaker test/validation data
│
├── lambda_package/
│   └── lambda_function.py      # AWS Lambda handler (inference + DynamoDB + SNS)
│
├── shap_global.png             # SHAP beeswarm — global feature importance
├── shap_bar.png                # SHAP bar chart — average feature impact
├── shap_waterfall.png          # SHAP waterfall — single-patient explanation
│
└── .gitignore
```

---

## 🔄 Workflow

### 1. Data Preparation

The raw Cleveland Heart Disease dataset is fetched from the UCI ML Repository, cleaned (missing values dropped), and binarized:

```python
df["target"] = (df["target"] > 0).astype(int)
df.dropna(inplace=True)
```

The processed data is split 80/20 (stratified) and uploaded to **Amazon S3**:

```
s3://health-risk-raghu/
  ├── data/raw/heart_processed.csv
  ├── data/train/train.csv
  └── data/test/test.csv
```

---

### 2. Model Training on SageMaker

An **XGBoost 1.5-1** built-in container is used for training on `ml.m5.large`:

| Hyperparameter | Value |
|---------------|-------|
| `objective` | `binary:logistic` |
| `num_round` | 100 |
| `max_depth` | 5 |
| `eta` | 0.2 |
| `subsample` | 0.8 |
| `eval_metric` | AUC |

The trained model artifact is saved to:
```
s3://health-risk-raghu/model-output/<job-name>/output/model.tar.gz
```

---

### 3. Endpoint Deployment

The model is deployed as a real-time **SageMaker endpoint**:

- **Endpoint Name:** `heart-disease-endpoint`
- **Instance Type:** `ml.m5.large`
- **Status:** `InService`

---

### 4. Lambda Function

The Lambda function (`health-risk-predictor`) serves as the orchestration layer:

```python
ENDPOINT   = "heart-disease-endpoint"
TABLE_NAME = "health-predictions"
TOPIC_ARN  = os.environ["TOPIC_ARN"]   # injected via environment variable
```

**Flow:**
1. Parse `patient_data` (list of 13 features) and `patient_id` from the event
2. Call SageMaker endpoint → get probability score
3. Classify: `HIGH_RISK` if score ≥ 0.5, else `LOW_RISK`
4. Write record to DynamoDB (`patient_id` + `timestamp` as composite key)
5. If `HIGH_RISK` → publish SNS alert to subscribed email
6. Return structured JSON response

**IAM Role:** `LambdaHealthRiskRole` with policies:
- `AmazonSageMakerFullAccess`
- `AmazonDynamoDBFullAccess`
- `AmazonSNSFullAccess`
- `AWSLambdaBasicExecutionRole`

---

### 5. Inference & Alerts

**DynamoDB Table:** `health-predictions`

| Attribute | Type | Role |
|-----------|------|------|
| `patient_id` | String | Partition Key |
| `timestamp` | String (ISO 8601) | Sort Key |
| `risk_score` | String (float) | Attribute |
| `prediction` | String | Attribute |
| `patient_data` | String | Attribute |

**SNS Topic:** `health-risk-alerts`  
High-risk patients trigger an email notification with:
- Patient ID
- Risk score
- Timestamp

---

### 6. Explainability with SHAP

A local XGBoost replica is trained on the same data to generate SHAP explanations:

| Plot | Description |
|------|-------------|
| `shap_global.png` | Beeswarm — shows how each feature drives predictions across all test patients |
| `shap_bar.png` | Bar chart — mean absolute SHAP value per feature |
| `shap_waterfall.png` | Waterfall — explains a single patient's prediction step by step |

**Top features by importance:** `thal`, `cp`, `ca`, `thalach`, `age`

---

## ☁️ AWS Services Used

| Service | Purpose |
|---------|---------|
| **Amazon S3** | Data storage (raw, train, test CSVs + model artifact) |
| **Amazon SageMaker** | Model training + real-time inference endpoint |
| **AWS Lambda** | Serverless inference orchestration |
| **Amazon DynamoDB** | Persistent prediction store |
| **Amazon SNS** | Real-time email alerts for high-risk patients |
| **Amazon IAM** | Role-based access control |

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (binary:logistic) |
| Threshold | 0.5 |
| Output | Probability score in [0, 1] |
| `HIGH_RISK` | score ≥ 0.5 |
| `LOW_RISK` | score < 0.5 |

---

## 📡 API Reference

### Lambda Input

```json
{
  "patient_id": "patient-001",
  "patient_data": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
}
```

> `patient_data` must be a list of 13 values in the feature order:  
> `[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]`

### Lambda Output

```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*"
  },
  "body": {
    "patient_id": "patient-001",
    "risk_score": 0.0113,
    "prediction": "LOW_RISK",
    "timestamp": "2026-04-02T08:46:13.772584"
  }
}
```

---

## 📈 SHAP Visualizations

| Plot | File |
|------|------|
| Global Feature Importance (Beeswarm) | `shap_global.png` |
| Average Feature Impact (Bar) | `shap_bar.png` |
| Single Patient Explanation (Waterfall) | `shap_waterfall.png` |

---

## 🚀 Getting Started

### Prerequisites

- AWS account with appropriate permissions
- Python 3.11
- Jupyter Notebook / JupyterLab

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ragha02/CardioSense.git
   cd CardioSense
   ```

2. **Set up the Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```

4. **Run the notebook**  
   Open `Untitled.ipynb` in Jupyter and execute all cells in order.

---

## 🛠 Environment Setup

The notebook uses the following key Python packages:

| Package | Purpose |
|---------|---------|
| `boto3` | AWS SDK — S3, SageMaker, Lambda, DynamoDB, SNS |
| `sagemaker` | SageMaker training + deployment SDK |
| `pandas` | Data manipulation |
| `scikit-learn` | Train/test split |
| `xgboost` | Local model for SHAP |
| `shap` | Explainability |
| `matplotlib` | SHAP plot rendering |

---
