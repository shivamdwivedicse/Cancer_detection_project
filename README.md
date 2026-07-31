<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=3000&pause=1000&color=F72585&center=true&vCenter=true&width=750&lines=Can+a+Machine+Spot+Cancer+Before+a+Doctor+Does%3F;Yes.+And+It+Takes+Less+Than+1+Second.;Welcome+to+the+Cancer+Detection+Project+%F0%9F%A9%BA" alt="Typing SVG" />

</div>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=Cancer%20Detection%20AI&fontSize=46&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Malignant%20vs%20Benign%20%E2%80%94%20Predicted%20by%20Machine%20Learning&descAlignY=55&descSize=18" />
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Accuracy-97.5%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">
</p>

---

## 🪝 Why This Project Matters

> **Every 2 minutes, someone is diagnosed with breast cancer.**
> Catching it early is the single biggest factor in survival.
> This project puts a trained Machine Learning model to work — reading tumor cell measurements and predicting **Malignant** or **Benign** in the blink of an eye.
>
> No lab coat. No waiting days for results. Just data, math, and a Streamlit app doing the heavy lifting. 🔬⚡

<div align="center">
  <img src="https://media.giphy.com/media/L1R1tvI9svkIWwpVYr/giphy.gif" width="400">
</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Live Demo](#-live-demo)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Models Trained](#-models-trained)
- [Final Model Results](#-final-model-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [App Preview](#-app-preview)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Connect With Me](#-connect-with-me)

---

## 🎯 About the Project

This project builds a Machine Learning model that studies the physical characteristics of a tumor — its **radius, texture, smoothness, symmetry**, and more — and predicts whether it is:

| Label | Meaning |
|-------|---------|
| 🔴 `1` | **Malignant** (cancerous) |
| 🟢 `0` | **Benign** (non-cancerous) |

The dataset used is the well-known **Breast Cancer Wisconsin Diagnostic Dataset**, containing 30 numeric features calculated from digitized images of tumor cell nuclei.

A user-friendly **Streamlit web app** is included, so anyone can enter values and instantly get a prediction — no coding needed.

---

## 🚀 Live Demo

<div align="center">

### 👉 [**Try the App Live Here**](https://cancerdetectionproject-rpkeeekh2rwhvwtajkxf5k.streamlit.app/) 👈

<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.png" width="100">

</div>

The app comes with **two modes**, so it's beginner-friendly and expert-friendly at the same time:

- 🟢 **Simple Mode** — fill in just the **10 most important values** → ~96.5% accuracy
- 🔵 **Advanced Mode** — fill in **all 30 values** for the most detailed prediction → ~97.5% accuracy

---

## ⚙️ How It Works

Raw Tumor Measurements → Data Cleaning → Feature Scaling → ML Model → Prediction

1. **Data Loading** — the Wisconsin dataset is loaded and cleaned (dropping ID and empty columns).
2. **Preprocessing** — features are scaled using `StandardScaler` so every measurement is compared fairly.
3. **Model Training** — multiple algorithms are trained and compared.
4. **Hyperparameter Tuning** — `GridSearchCV` finds the best settings for Logistic Regression.
5. **Threshold Analysis** — different probability cut-offs are tested to reduce dangerous **False Negatives** (missing an actual cancer case).
6. **Deployment** — the final model is saved with `joblib` and served through a **Streamlit** web app.

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,sklearn,pandas,git,github,vscode" />
</p>

| Category | Tools Used |
|----------|-----------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn |
| **Web App** | Streamlit |
| **Model Saving** | Joblib |

---

## 🧠 Models Trained

Several algorithms were trained and compared to find the best performer:

- ✅ Logistic Regression *(final chosen model)*
- 🌲 Random Forest
- 🧮 Support Vector Machine (SVM)
- 📍 K-Nearest Neighbors (KNN)
- 📈 Gradient Boosting

After comparison, **Logistic Regression** came out on top for its balance of accuracy, speed, and interpretability.

---

## 🏆 Final Model Results

LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')

**Best hyperparameter found using GridSearchCV:** C = 0.5

| Metric | Score |
|--------|-------|
| ✅ Test Accuracy (Full Model) | **~97.5%** |
| ✅ Test Accuracy (Simple 10-feature Model) | **~96.5%** |
| ✅ Cross Validation Mean Score | **94.90%** |

**Confusion Matrix:** [[70, 1], [1, 42]]

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
</p>

---

## 📂 Project Structure

Cancer_detection_project/
├── app.py                     (Streamlit web app)
├── train_final_model.py       (Model training script)
├── Cancer_project.ipynb       (Full notebook - EDA + experiments)
├── Cancer_Data.csv            (Dataset)
├── cancer_model.pkl           (Trained full model, 30 features)
├── scaler.pkl                 (Scaler for full model)
├── cancer_model_simple.pkl    (Trained simple model, 10 features)
├── scaler_simple.pkl          (Scaler for simple model)
├── requirements.txt           (Python dependencies)
└── README.md                  (You're here!)

---

## 🏁 Getting Started

**1️⃣ Clone the repository**

git clone https://github.com/shivamdwivedicse/Cancer_detection_project.git
cd Cancer_detection_project

**2️⃣ Install the dependencies**

pip install -r requirements.txt

**3️⃣ Run the app**

streamlit run app.py

**4️⃣ Open your browser**

Streamlit will automatically open the app at: http://localhost:8501

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/213910845-af37a709-8995-40d6-be59-724526e3c3d7.gif" width="400">
</p>

---

## 🖥️ App Preview

| Simple Mode 🟢 | Advanced Mode 🔵 |
|---|---|
| Fill 10 key values → instant prediction | Fill all 30 values → deep, detailed prediction |
| Great for a quick check | Great for full analysis |

> ⚠️ **Disclaimer:** This tool is built for **educational and demo purposes only**. It is **not** a certified medical diagnostic device. Always consult a qualified doctor for real medical decisions.

---

## 🔮 Future Improvements

- [ ] Add Deep Learning model (Neural Network) comparison
- [ ] Add SHAP/feature-importance visual explanations
- [ ] Add image-based tumor detection (CNN on histopathology images)
- [ ] Deploy on Hugging Face Spaces as a backup mirror
- [ ] Add user authentication + prediction history

---

## 👨‍💻 Author

<div align="center">

### **Shivam Dwivedi**

<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="150">

*"Turning data into decisions, one model at a time."* 🚀

</div>

---

## 🔗 Connect With Me

<p align="center">
  <a href="https://github.com/shivamdwivedicse">
    <img src="https://img.shields.io/badge/GitHub-shivamdwivedicse-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://www.linkedin.com/in/shivam-dwivedi-27661a395">
    <img src="https://img.shields.io/badge/LinkedIn-Shivam%20Dwivedi-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="mailto:shivamdwivedicse20919@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
</p>

<div align="center">

### ⭐ If this project helped you, don't forget to **star the repo**!

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" />

</div>
