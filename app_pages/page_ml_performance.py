import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
import pickle  # Import the pickle module

# from src.machine_learning.evaluate_clf import load_test_evaluation

def load_test_evaluation(version):
    # Assuming evaluation.pkl is a DataFrame saved in pickle format
    # Adjust the path as necessary
    evaluation_path = f"outputs/{version}/evaluation.pkl"
    try:
        with open(evaluation_path, 'rb') as file:
            evaluation = pickle.load(file)
        return evaluation
    except FileNotFoundError:
        print(f"File not found: {evaluation_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found

def page_ml_performance_metrics():
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/v1/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))

def main():
    st.title("ML Performance")
    page_ml_performance_metrics()

if __name__ == "__main__":
    main()
