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
    This bar chart demonstrates balanced class distribution across the train, validation, 
    and test sets, ensuring an equal number of healthy and powdery mildew images in each category. 
    Such balance is crucial for the model to learn effectively without bias towards any particular class,
    fostering equitable distinction between the two classes.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    labels_distribution = plt.imread(f"outputs/{version}/pie_chart.png")
    st.image(labels_distribution, caption='Data Distribution on Train, Validation and Test Sets')
    explanation_text = """
    This pie chart visually represents the dataset, which consists of 4208 image files,
    equally divided between 2104 healthy and 2104 powdery mildew-labeled images.
    The dataset underwent partitioning into Train, Validation, 
    and Test sets with respective ratios of 0.7, 0.1, and 0.2.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
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
    The graph illustrates the learning cycle of the ML model through two plots showcasing accuracy and loss metrics during training. 
    Analysis of these plots suggests a normal learning curve, indicating that the model neither suffers from overfitting nor underfitting. 
    This conclusion is drawn from observing a balance between the model's ability to generalize well to unseen data and its capacity to learn from the training data effectively. 
    The accuracy plot demonstrates a gradual increase over epochs, suggesting the model's improvement in correctly predicting outcomes. 
    Concurrently, the loss plot exhibits a steady decline, indicating diminishing errors as the model learns. 
    Overall, these trends suggest a healthy training process where the model achieves a desirable balance between accuracy and generalization.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")
    

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    explanation_text = """
    The plot illustrates the evaluation metrics of the trained ML model on the test dataset,
    with a reported loss value of 0.04569917 and an accuracy value of 0.9917.
    The loss value signifies the average deviation of predicted values from true values,
    with lower values indicating better performance in binary classification tasks.
    The accuracy value of 0.9917 indicates that the model correctly classified approximately 99.17% of instances in the test dataset.
    These metrics suggest strong performance and robust generalization of the model to new, unseen data.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    st.write("### Classification Report")
    labels_distribution = plt.imread("outputs/v1/classification_report.png")
    st.image(labels_distribution, caption='')
    explanation_text = """
    The classification report offers comprehensive metrics for each class in the model's predictions,
    including "healthy" and "powdery mildew." For the "healthy" class, precision, recall,
    and F1-score metrics are exceptionally high, reflecting the model's capability to
    accurately identify instances of healthy leaves while minimizing false positives and
    false negatives. Similarly impressive metrics are observed for the "powdery mildew" class.
    The model demonstrates an overall accuracy of 99%, indicating its proficiency in making
    precise predictions across both classes. Furthermore, the macro average and weighted average
    metrics provide consolidated performance evaluations across all classes,
    reaffirming the model's robust performance.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    st.write("### Confusion Matrix")
    labels_distribution = plt.imread("outputs/v1/confusion_matrix.png")
    st.image(labels_distribution, caption='')
    explanation_text = """
    The confusion matrix is divided into four segments, each representing different prediction 
    outcomes: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). 
    TP and TN represent correct predictions, whereas FP and FN denote incorrect ones. 
    The accuracy metric, at 99%, indicates the percentage of correct predictions made by the model.
    A high accuracy coupled with a low loss suggests the model is effectively making accurate predictions.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")

    st.write("### ROC Curve")
    labels_distribution = plt.imread("outputs/v1/roc_curve.png")
    st.image(labels_distribution, caption='')
    explanation_text = """
    The ROC curve serves as a performance metric, indicating the model's ability to effectively 
    differentiate between classes by making precise predictions. 
    It is constructed by plotting the true positive rate (TPR) against the false positive rate (FPR).
    The TPR represents the proportion of accurately predicted observations,
    where a leaf predicted as healthy is indeed healthy. Conversely, the FPR signifies the ratio of
    incorrectly predicted instances, such as when a leaf classified as healthy is actually affected.
    By analyzing these rates, the ROC curve provides insights into the model's discrimination
    capabilities across different classification thresholds.
    This evaluation is crucial for understanding the model's effectiveness in distinguishing between
    classes and making informed decisions regarding its performance.
    """
    st.markdown(f"<div style='background-color: #caccd1; padding: 10px; border-radius: 5px;'>{explanation_text}</div>", unsafe_allow_html=True)
    st.write("---")



def main():
    st.title("ML Performance")
    page_ml_performance_metrics()

if __name__ == "__main__":
    main()
