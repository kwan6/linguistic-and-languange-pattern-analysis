# Linguistic and Language Patterns Analysis w/ 3 Different Algorithms

# Output Example

  - Accuracy: 0.82 (IndoBERT)
  - F1-score: 0.74
  
## Objectives

  - Analyze linguistic patterns in Indonesian clickbait headlines
  - Compare performance between classical ML models and transformer-based models
  - Evaluate model effectiveness using standard metrics (Accuracy, Precision, Recall, F1-score)

## Methods

### Machine Learning Models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)

### Deep Learning:
  - IndoBERT (Pretrained Transformer Model for Indonesian Language)

## Dataset

  - CLICK-ID: A Novel Dataset for Indonesian Clickbait Headlines
  - Contains labeled news headlines from multiple Indonesian media sources

## Features

  - Data preprocessing and cleaning
  - TF-IDF vectorization (for ML models)
  - Transformer-based fine-tuning (IndoBERT)
  - Evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score

## Results (Summary)

  - Machine Learning models achieved competitive baseline performance
  - IndoBERT demonstrated better contextual understanding and improved classification results

## Tech Stack

  - Python
  - Scikit-learn
  - HuggingFace Transformers
  - PyTorch
  - Pandas, NumPy

## Project Structure

   clickbait-nlp/
    │── data/
    │── src/
    │ ├── data_loader.py
    │ ├── model_ml.py
    │ ├── model_indobert.py
    │── main.py
    │── requirements.txt

## Notes
  
## 📌 Notes

This project is developed as part of a research study on NLP-based clickbait detection in Indonesian language.
