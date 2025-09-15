# 📩 SMS Spam Classifier  

A simple **Streamlit web application** that uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to classify SMS messages as **Spam** or **Ham (Not Spam)**.  


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Downloads](https://img.shields.io/pypi/dm/sms-spam-classifier)](https://pypi.org/project/sms-spam-classifier/)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/subhadipsinha722133/SMS-Spam-Classifier)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/commits/main)
[![Coverage Status](https://img.shields.io/codecov/c/github/subhadipsinha722133/SMS-Spam-Classifier)](https://codecov.io/gh/subhadipsinha722133/SMS-Spam-Classifier)
[![Release](https://img.shields.io/github/v/release/subhadipsinha722133/SMS-Spam-Classifier)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/releases)

[![Open Issues](https://img.shields.io/github/issues/subhadipsinha722133/SMS-Spam-Classifier)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/subhadipsinha722133/SMS-Spam-Classifier)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/pulls)
[![Stars](https://img.shields.io/github/stars/subhadipsinha722133/SMS-Spam-Classifier?style=social)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/stargazers)
[![Forks](https://img.shields.io/github/forks/subhadipsinha722133/SMS-Spam-Classifier?style=social)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/forks)
[![Contributors](https://img.shields.io/github/contributors/subhadipsinha722133/SMS-Spam-Classifier)](https://github.com/subhadipsinha722133/SMS-Spam-Classifier/graphs/contributors)

[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)


---
## 📺 Live Demo
🔗[Demo](https://subhadipsinha722133-6czxjynlrcg24rua5bwakx.streamlit.app/)

---

## 🚀 Features  
- Clean and interactive UI built with **Streamlit**  
- Preprocessing using **NLTK** (tokenization, stopword removal, stemming)  
- Machine Learning with **Scikit-learn**  
- Real-time prediction of SMS spam messages  

---

## 🛠️ Tech Stack  
- **Python 3.8+**  
- **Streamlit** - Web app framework  
- **NLTK** - Natural Language Toolkit for text preprocessing  
- **Scikit-learn** - Machine learning models  
---

## 📂 Project Structure  
SMS-Spam-Classifier/ <br>
|-- ms-spam-detection.ipynb # Apply all algorithm<br>
│-- main.py # Streamlit app script <br>
│-- model.pkl # Trained ML model <br>
│-- vectorizer.pkl # Fitted text vectorizer <br>
│-- requirements.txt # Project dependencies <br>
│-- README.md # Project documentation <br>



---

## ⚙️ Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/subhadipsinha722133/SMS-Spam-Classifier.git
   cd SMS-Spam-Classifier
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate     # On Mac/Linux
venv\Scripts\activate        # On Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
# ▶️ Usage
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Then open the local URL provided in your browser (usually http://localhost:8501/).

model.pkl file very large 119mb

# 📊 Model Workflow
Text preprocessing with NLTK

Feature extraction using TF-IDF Vectorizer<br>

- Classification with 
  - from sklearn.linear_model import LogisticRegression
  - from sklearn.svm import SVC
  - from sklearn.naive_bayes import MultinomialNB
  - from sklearn.tree import DecisionTreeClassifier
  - from sklearn.neighbors import KNeighborsClassifier
  - from sklearn.ensemble import RandomForestClassifier
  - from sklearn.ensemble import AdaBoostClassifier
  - from sklearn.ensemble import BaggingClassifier
  - from sklearn.ensemble import ExtraTreesClassifier
  - from sklearn.ensemble import GradientBoostingClassifier
  - from xgboost import XGBClassifier  
(choose your trained model)


- **from sklearn.neighbors import KNeighborsClassifier** <br>
  **Accuracy 0.9988931931377975<br>
  Precision 0.9978118161925602**

**Output: Spam or Not Spam**



# 🤝 Contributing
Contributions are welcome! Feel free to fork this repo and submit a pull request.

