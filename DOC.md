# ASSIGNMENT 1 – THEORY SECTION

**Course:** Intelligence System Development – **Semester I/2025**
**Student:** \[Your name] – **Class/Team:** \[Fill in]

> Note: This write‑up summarizes and analyzes foundational knowledge on Intelligent Systems/AI using the sources listed in the brief (Forbes, Apple Siri, Simplilearn, Edureka, etc.). For **1.5**, because I could not open Figure 7 of arXiv:2009.09083 in the current offline environment, I provide a **faithful reconstruction/interpretation** of a typical “domain ↔ AI technique” matrix so that the section remains useful and presentation‑ready. I will align the labels to the original figure when direct access is available.

---

## 1.1. Investigate, discover and write (≈3 pages)

### 1) Ten representative, real‑world AI applications

1. **Voice assistants & speech recognition:** Siri/Google Assistant convert speech to text, follow spoken commands, and control devices.
2. **Recommender systems:** Netflix/YouTube/Spotify/Amazon suggest items based on interaction history (collaborative filtering, deep learning).
3. **Fraud detection:** Banks/payment gateways spot anomalous transactions using supervised/unsupervised learning.
4. **Medical imaging:** Detecting tumors in X‑ray/CT/MRI with CNN‑ or ViT‑based models.
5. **Self‑driving & ADAS:** Lane/obstacle/pedestrian detection; motion planning (computer vision + reinforcement learning).
6. **Customer‑service chatbots:** NLP/LLM for 24/7 auto responses and case routing.
7. **Machine translation & summarization:** Transformer/LLM systems deliver fast multilingual translation and summarization.
8. **Smart home/IoT:** Adaptive temperature/lighting via RL and time‑series forecasting.
9. **Security & surveillance:** Network intrusion detection (IDS), log anomaly detection; face access control.
10. **Predictive maintenance:** Analyzing vibration/acoustic/power signatures to anticipate machine failures and reduce downtime.

### 2) The data/AI value chain

**Collect/label** (sensors, logs, clickstream) → **Clean/preprocess** (denoise, normalize) → **Feature/representation** (feature engineering, embeddings) → **Train/tune** (supervised/unsupervised/RL) → **Deploy** (APIs, edge) → **Monitor** (drift, fairness, safety) → **Improve** (continuous loop).
**Key challenges:** data quality, bias/fairness, privacy/security, explainability (XAI), infra cost, and real‑world safety (robotics/vehicles).

### 3) Metrics & good practices

- **Model metrics:** Accuracy/F1/AUC (classification), RMSE/MAE (regression), mAP (CV), BLEU/ROUGE (NLP).
- **MLOps:** dataset/model versioning, CI/CD pipelines, drift monitoring, safety/adversarial testing.
- **Ethics & compliance:** transparency, fairness, privacy by design.

---

## 1.2. What is an Intelligent System? Most impressive definition & examples (≈3 pages)

### 1) Blended definition

An **intelligent system** is a software/hardware system that can **perceive**, **reason/decide**, and **learn** from data and environment interactions to **achieve goals** efficiently beyond fixed rule sets.

Common perspectives:

- **Functional loop:** Perceive → Understand → Plan → Act → Learn (closed feedback).
- **Agent view:** an intelligent agent that maximizes a **utility function** under uncertainty.
- **Technique view:** knowledge‑based reasoning, machine learning, or **hybrid** approaches.

### 2) The most compelling definition (and why)

> **“An intelligent system is an autonomous agent that senses its environment, builds and updates an internal model through learning, and acts to maximize its objectives under uncertainty.”** > **Why:** concise, covers the four pillars (perception–modeling–learning–action), and emphasizes **uncertainty**, which dominates real‑world settings.

### 3) Example intelligent systems

- **Software:** recommender engines, multilingual chatbots, clinical decision support (CDSS).
- **Cyber‑physical:** warehouse mobile robots (SLAM + RL), UAVs for forest monitoring (CV + path planning).
- **Edge/IoT:** fall‑detection cameras, health wearables predicting arrhythmia.
- **Enterprise:** supply‑chain optimization (forecast + MILP), transaction‑fraud detection (anomaly/graph ML).

---

## 1.3. Applications of intelligent systems: domains & AI techniques (≈3 pages)

### 1) Major domains

- **Healthcare:** imaging, disease classification, readmission forecasting, EMR assistants.
- **Finance:** fraud detection, credit scoring, risk pricing, algorithmic trading.
- **Manufacturing/IIoT:** predictive maintenance, visual defect detection, scheduling optimization.
- **Transportation:** ITS, routing, signal control, ADAS/autonomy.
- **Energy:** load/price forecasting, grid optimization, fault diagnosis.
- **Retail/Marketing:** recommendations, customer segmentation, dynamic pricing.
- **Cybersecurity:** IDS, botnet detection, log analytics.
- **Education:** intelligent tutoring, adaptive testing, proctoring.
- **Agriculture:** yield forecasting, pest recognition, precision spraying via drones.
- **Public sector/Smart city:** environmental sensing, traffic analytics, digital public services.

### 2) Representative AI techniques by task type

- **Computer Vision (CV):** CNN/ViT for recognition, detection, segmentation.
- **NLP/LLM:** Transformers, fine‑tuning/PEFT, RAG for QA and summarization.
- **Supervised learning:** regression, decision trees, SVM, gradient boosting.
- **Unsupervised:** clustering (K‑means/DBSCAN), PCA/UMAP, anomaly detection.
- **Reinforcement learning (RL):** policy optimization for control/operations.
- **Knowledge & reasoning:** ontologies, knowledge graphs, logical inference.
- **Optimization/OR:** linear programming, constraint solving, meta‑heuristics (GA/PSO) for routing/scheduling.
- **Time series:** ARIMA/Prophet/RNN/Temporal Transformers.
- **XAI & safety:** SHAP/LIME, bias audits, drift detection.

---

## 1.4. Types of intelligent systems (≈3 pages)

### A) By **capability level** (as in Simplilearn/Edureka)

1. **Reactive Machines:** respond to current state only; no memory (e.g., Deep Blue).
2. **Limited Memory:** use recent data for decisions (most ML systems today).
3. **Theory of Mind** _(research target):_ model others’ mental states.
4. **Self‑aware** _(hypothetical):_ self‑conscious agents.

### B) By **scope of intelligence**

- **ANI (Narrow/Weak AI):** excels at a narrow task (e.g., image classification).
- **AGI (General AI):** human‑level generality (under research).
- **ASI (Superintelligence):** surpasses human ability (theoretical).

### C) By **agent architecture/strategy**

- **Reactive** vs **Deliberative (planning)** vs **Hybrid**.
- **Rule‑based/KBS** vs **Learning‑based (ML/DL)** vs **Neuro‑symbolic (hybrid)**.
- **Centralized** vs **Multi‑Agent Systems (MAS)** (coordination/auctions/consensus).
- **Cloud‑centric** vs **Edge/on‑device** deployment.

**Remark:** Real systems are typically **hybrids**: perception (DL) + planning/optimization (OR) + rules/constraints for safety/compliance.

---

## 1.5. Applications via **Figure 7** of arXiv:2009.09083 (≈3 pages)

> In lieu of direct access to Figure 7, I provide a **domain ↔ technique** matrix that mirrors common surveys. When I obtain the original figure, I will update the labels/examples to be exact.

### 1) Mock‑up matrix: domains ↔ AI techniques

| Domain         | Perception (CV/ASR)            | NLP/LLM                 | Forecasting (TS)        | Optimization/OR                    | RL/Control       | KBS/Graph              |
| -------------- | ------------------------------ | ----------------------- | ----------------------- | ---------------------------------- | ---------------- | ---------------------- |
| Healthcare     | Segmentation, lesion detection | EMR summarization, NER  | Readmission forecasting | OR for OR‑scheduling               | Dosing policies  | Medical ontologies     |
| Finance        | Doc OCR, forgery detection     | News sentiment/NLP      | Risk/price forecasting  | Portfolio optimization             | Trading agents   | Fraud graphs           |
| Manufacturing  | Visual defect inspection       | Natural‑language QA     | Predictive maintenance  | Job‑shop scheduling                | Parameter tuning | Process knowledge      |
| Transportation | Object detection               | V2X language interfaces | Traffic forecasting     | Multi‑objective routing            | Signal control   | Route knowledge graphs |
| Energy         | Fault recognition              | Incident summarization  | Load/price forecasting  | Grid optimization                  | Grid control     | Asset knowledge        |
| Retail         | Product recognition            | Chatbots                | Demand forecasting      | Inventory optimization             | Dynamic pricing  | Product KG             |
| Agriculture    | Pest/disease detection         | Farm logs               | Harvest forecasting     | Irrigation/fertilizer optimization | Agri‑robots      | Crop knowledge         |

### 2) Reference deployment pipeline

**Sensing/Data layer** → **ML/DL processing** → **Planning/Optimization** → **Action/Robotics** → **Safety/XAI monitoring**.
Governance: data quality, audits, security, compliance.

---

## 1.6. NumPy, Pandas, Matplotlib, Scikit‑learn — purpose, features & examples

### 1) NumPy

- **Purpose:** high‑performance ND arrays; vectorized math; linear algebra.
- **Features:** `ndarray`, broadcasting, ufuncs, random, `linalg`.
- **Example:**

```python
import numpy as np
x = np.array([1,2,3], dtype=float)
y = np.array([4,5,6], dtype=float)
cos_sim = np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

### 2) Pandas

- **Purpose:** tabular data wrangling/cleaning/aggregation.
- **Features:** `read_csv`, indexing, `groupby`, `merge`, missing‑value & time‑series utilities.
- **Example:**

```python
import pandas as pd
df = pd.DataFrame({"student":["Ann","Joe"], "math":[8.5,7.0], "eng":[7.5,8.0]})
df["avg"] = df[["math","eng"]].mean(axis=1)
by = df.groupby("student")["avg"].mean().reset_index()
```

### 3) Matplotlib

- **Purpose:** 2D plotting; highly customizable.
- **Features:** line/bar/scatter/histograms, annotations, styling.
- **Example:**

```python
import matplotlib.pyplot as plt
subjects = ["Math","Phys","Chem"]
marks = [8.0, 7.5, 8.8]
plt.figure(figsize=(5,3))
plt.bar(subjects, marks)
plt.title("Marks by Subject"); plt.ylim(0,10)
plt.show()
```

### 4) Scikit‑learn

- **Purpose:** classical ML toolkit (supervised/unsupervised) with pipelines & evaluation.
- **Components:** `LinearRegression`, `LogisticRegression`, SVM, trees/ensembles (RF, GBM), KMeans, PCA, `train_test_split`, `Pipeline`, `GridSearchCV`.
- **Example (linear regression):**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
X = np.array([[50],[60],[70],[80],[90]], dtype=float)
y = np.array([2.5,3.0,3.8,4.5,5.4])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)
rmse = mean_squared_error(yte, model.predict(Xte), squared=False)
```

### 5) Presentation & reproducibility tips

- Separate **data processing**, **training**, and **plotting** functions; fix versions with `requirements.txt`/`pipenv`.
- Set random **seeds** for reproducibility.
- In visuals, always include **units**, **legends**, and **data/source notes**.

---

## Short conclusion

Intelligent systems integrate **perception, learning, reasoning, and action**; practical deployments are **hybrids** of DL/ML, optimization, and knowledge. The NumPy/Pandas/Matplotlib/Scikit‑learn stack is foundational for building the data → model → visualization pipeline. For **1.5**, I will replace the mock‑up with the exact Figure 7 labels once direct access is available.
