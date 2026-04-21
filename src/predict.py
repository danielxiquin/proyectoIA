import pickle
import sys
import os

def load_model(model_path="model/naive_bayes_model.pkl"):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

def classify(text, model_data):
 
    preprocessor = model_data['preprocessor']
    classifier = model_data['classifier']
    
    tokens = preprocessor.preprocess(text)
    
    predicted_class = classifier.predict_one(tokens)
    probabilities = classifier.predict_proba(tokens)
    
    return predicted_class, probabilities

