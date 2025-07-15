# Fake News Detection Using Machine Learning

This project is a machine learning-based application that detects whether a given news article is **real** or **fake**. It uses **Natural Language Processing (NLP)** techniques to classify news text into two categories using a clean and interpretable model pipeline.

## Why I Built This

Fake news is a critical challenge in today’s digital world. This project helped me gain hands-on experience in working with real-world text data, cleaning and vectorizing it, and applying classification models — all while building a useful fake news detector.

## Dataset

- **Files Used**:
  - `fake.csv`: Contains fake news articles (title, text)
  - `real.csv`: Contains legitimate news articles

These files were **manually merged and labeled** during preprocessing to create a clean training dataset for the model.

If dataset is too large for GitHub, download from:  
🔗 [Google Drive Dataset Link](https://drive.google.com/drive/folders/1LT-5ZGIyRzIoDQ34vMFPkaVubDzXdn2p?usp=sharing)

## Model Pipeline Overview

All logic is implemented in the notebook `Fake_News_Detection.ipynb`.

### 🔧 Steps Followed:

1. **Data Loading**: Load `fake.csv` and `real.csv`, add a `label` column (`FAKE`/`REAL`)
2. **Preprocessing**:
   - Remove stopwords, lowercase conversion, remove punctuation
   - Tokenization and stemming using NLTK
3. **Vectorization**:
   - Used `TfidfVectorizer` (or `CountVectorizer`) to convert text to numeric format
4. **Model Training**:
   - Trained a `Multinomial Naive Bayes` model
   - Used train-test split and evaluated model with accuracy, confusion matrix, and F1 score
5. **Prediction Function**:
   - Takes raw news text input and returns predicted label

## Example

**Input:**
> "NASA discovers new habitable exoplanet in nearby solar system!"

**Output:**
> REAL

**Input:**
> "Government replacing doctors with AI robots in all hospitals."

**Output:**
> FAKE

## Tech Stack

- Python  
- Pandas & NumPy  
- NLTK  
- Scikit-learn  
- TfidfVectorizer  
- Multinomial Naive Bayes

## Future Enhancements

- Fine-tune the model using hyperparameter tuning and n-gram text features  
- Experiment with advanced classifiers (e.g., Logistic Regression, PassiveAggressiveClassifier)  
- Add more preprocessing steps like removing URLs, numbers, and HTML tags  
- Save and load the trained model and vectorizer using `pickle`  
- Build an interactive web interface using **Streamlit**  
- Deploy the project as an end-to-end web app using **Streamlit Cloud** or **Render**  
- Accept real-time news article input and display classification results live  
- Extend the system to detect satire or biased news using additional labels


