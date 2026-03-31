# 🔐 Phishing URL Detection System using Deep Learning

## 📌 Course Details
- **Course Name:** Fundamentals of AI and ML  
- **Course Code:** CSA2001  
- **Faculty:** Prof. J. Manikandan Sir
- **Student Name:** Sarthak Jindal  
- **Register Number:** 25BCE10189  
- **Slot:** B11 + B12 + B13  

---

## 🚀 Project Overview

This project aims to detect **phishing websites** using deep learning models. Phishing attacks are a major cybersecurity threat where attackers trick users into revealing sensitive information via malicious URLs.

The system uses multiple neural network architectures to classify URLs as:
- ✅ Legitimate  
- ⚠️ Suspicious  
- ❌ Phishing  

---

## 🎯 Objectives

- Build a machine learning system for phishing detection
- Compare multiple deep learning architectures
- Evaluate models using accuracy, ROC curve, and confusion matrix
- Provide an interactive tool for real-time prediction

---

## 🧠 Models Implemented

1. CNN + BiGRU ⭐ (Best Performing)
2. CNN + BiLSTM
3. LSTM + GRU
4. BiGRU

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve (AUC)
- Confusion Matrix

---

## 📂 Project Structure

📁 Fundamentals Of AI and ML- VITyarthi
│── main.py
│── Phising_Training_Dataset.csv
│── model_results.png
│── README.md
│── Project Report.pdf
│── requirements.txt


---

## ⚙️ Installation & Setup

### Step 1: Clone Repository
```bash
git clone <your-repo-link>
cd phishing-detection
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Run Project
```bash
python main.py
```

---

## 📈 Output

* Model accuracy comparison graph
* ROC curve visualization
* Confusion matrices
* Best model selection
* Interactive phishing detector

---
## 🖥️ Interactive Detector

After training, you can test URLs manually by entering feature values:

Example inputs:
- SSL Final State: 1
- URL Anchor Score: 0
- Web Traffic Score: -1

Output:
SAFE / SUSPICIOUS / PHISHING

---
## 🏆 Results

* Best Model: CNN + BiGRU
* Accuracy: ~90%
* AUC: ~0.96

---
## ⚠️ Challenges Faced
+ Dataset imbalance
+ Feature encoding understanding
+ Model tuning with limited epochs
+ Visualization clarity

---
## 📚 Learnings
* Deep learning model comparison
* ROC & confusion matrix interpretation
* Real-world cybersecurity application
* TensorFlow model building

---

## 🔮 Future Improvements
- Use real-time URL scraping
- Deploy as web application
- Increase dataset size
- Improve model accuracy

---

## 📌 Conclusion

This project demonstrates how AI/ML can be applied to solve real-world cybersecurity problems like phishing detection efficiently.

---

## ⭐ Author
* Sarthak Jindal
* Reg No: 25BCE10189

---