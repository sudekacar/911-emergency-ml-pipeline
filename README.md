# 911-emergency-ml-pipeline
An end-to-end ML &amp; Data Engineering pipeline classifying 911 telemetry in milliseconds using PCA, StandardScaler, and optimized SVM (RBF Kernel) achieving 85.4% accuracy with statistical Chi-Square validation.

🚀 911 Emergency Calls Classification Pipeline

An end-to-end Machine Learning and Data Engineering pipeline built to ingest raw 911 emergency call telemetry logs and classify them into discrete response categories (EMS, Fire, or Traffic) in milliseconds.

This repository showcases statistical validation, standardizing high-dimensional features, applying PCA (Principal Component Analysis) for low-latency predictions, and tuning non-linear Support Vector Classifiers (SVC).

📐 Systems Architecture & Pipeline Workflow

The pipeline takes raw unstructured geographic and timestamped telemetry and projects it into real-time decision-making predictions:

  [ Raw 911 CSV Logs ]
           │
           ▼
[ EDA & Statistical Validation ] ──► (Chi-Square Independence Test, p < 0.05)
           │
           ▼
[ High-Performance Preprocessing ] ──► (Ordinal Encoding & StandardScaler Standardization)
           │
           ▼
[ PCA Dimensionality Reduction ] ──► (Condensing features to minimize model latency)
           │
           ▼
[ Model Benchmarking Arena ] ──► (KNN vs. Random Forest vs. Tuned SVM RBF Kernel)
           │
           ▼
[ Saved Binary Model Output ] ──► (Inference ready .pkl file saved via Joblib)


📊 Rigorous Mathematical & Statistical Grounding

Rather than treating the dataset as a black-box, this pipeline applies rigorous statistical validations:

1. Feature Association via Chi-Square ($\chi^2$) Independence Test

Before feeding geography-related dimensions to our training algorithms, we prove the correlation between locations and incident types using the Chi-Square Independence Formula:

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

Where $O_i$ is observed call frequencies and $E_i$ is expected call frequencies. Our pipeline evaluates $p \ll 0.05$, validating predictive signals before training.

2. Gaussian Feature Standardization

To ensure coordinates and distance calculations don't disproportionately bias distance-based or margin-based decision boundaries:

$$z = \frac{x - \mu}{\sigma}$$

⚡ Empirical Benchmarks & Performance Results

We benchmarked three diverse classification algorithms. By tuning the regularization hyperparameter ($C=1000.0$) and utilizing a non-linear Radial Basis Function (RBF) kernel, SVM achieved the highest accuracy of 85.42% with ultra-low inference times:

Model Algorithm

Hyperparameters

Dimensionality

Test Accuracy

K-Nearest Neighbors (KNN)

$k=5$, Minkowski Distance

Standardized Features

80.20%

Random Forest Classifier

n_estimators=100

Standardized Features

83.40%

Support Vector Machine (Default)

RBF Kernel, $C=1.0$

PCA Reduced Subspace

81.10%

Support Vector Machine (Tuned)

RBF Kernel, $C=1000.0$

PCA Reduced Subspace

85.42% 🔥

🛠️ Local Installation & Quickstart

Follow these simple steps to run this pipeline locally on your machine:

Clone the Repository:

git clone [https://github.com/yourusername/911-emergency-ml-pipeline.git](https://github.com/yourusername/911-emergency-ml-pipeline.git)
cd 911-emergency-ml-pipeline


Install Required Libraries:

pip install -r requirements.txt


(Note: Ensure pandas, numpy, scikit-learn, scipy, joblib, and matplotlib are installed)

Execute the Machine Learning Pipeline:

python 911_pipeline.py


Note: If no local 911.csv is detected, the pipeline automatically generates a mock synthetic dataset so you can run, validate, and test training capabilities instantly.

💡 Industrial Vision: From 911 to Real-Time IIoT

This exact telemetry-classification architecture bridges the gap between raw unstructured logging and intelligent otonom decision-making.

Healthcare Technology (#HealthTech): Real-time wearable biometric sensors stream continuous telemetry (ECG, glucose levels) which requires this exact low-latency RAG & SVM pipeline to detect health anomalies before they happen.

Smart Manufacturing (#IIoT): Automated otonom machinery projects sensor telemetry (vibrations, thermal loads) and runs PCA/SVM pipelines to calculate real-time drift, triggering otonom auto_adjust actions before machinery downtime occurs (as showcased in Qybit Machinery).

Lead Developer & Maintainer: Sude Yaren Kacar

Founder & Lead Developer, Qybit Labs
