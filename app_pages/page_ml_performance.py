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
    labels_distribution = plt.imread(f"outputs/{version}/bar_chart.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    explanation_text = """
    This Python script generates a bar chart using Plotly Express, depicting the distribution of labels ('healthy' or 'powdery_mildew') across different dataset splits ('train,' 'test', and 'validation'). The chart displays the count of each label within each dataset split and automatically labels the bars with their respective counts.
    """
    st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    labels_distribution = plt.imread(f"outputs/{version}/pie_chart.png")
    st.image(labels_distribution, caption='Data Distribution on Train, Validation and Test Sets')
    explanation_text = """
    The code generates a pie chart using Plotly Express to visualize the distribution of a dataset across different sets (e.g., train, test, validation). It creates the chart based on provided set names and their corresponding counts, displaying each set's proportion using different colors. 
    """
    st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    explanation_text = """
    A model history plot visually illustrates the performance of a machine learning model throughout the training and validation phases. It consists of two curves: one for the training dataset (typically shown in blue) and another for the validation dataset (often depicted in orange). The y-axis represents the chosen performance metric, such as accuracy or loss, while the x-axis indicates the number of training epochs. Effective interpretation of these plots enables practitioners to assess model learning dynamics, identify potential overfitting, and optimize model training strategies.    """
    st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    explanation_text = """
    The plot illustrates the performance metrics of a trained ML model on the test dataset. It includes two crucial metrics: loss and accuracy, with the former measuring prediction deviations and the latter denoting the proportion of correctly classified instances. A low loss value (0.0026) coupled with a high accuracy value (0.9976) indicates the model's robustness and its ability to generalize well to unseen data.
    """
    st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)

def main():
    st.title("ML Performance")
    page_ml_performance_metrics()

if __name__ == "__main__":
    main()
