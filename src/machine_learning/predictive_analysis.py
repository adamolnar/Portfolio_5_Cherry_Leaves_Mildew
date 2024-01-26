import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import cv2
from PIL import Image
from src.data_management import load_pkl_file
import joblib

# Function to plot prediction probabilities
def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """
    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'Parasitized': 0, 'Uninfected': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn'
    )
    st.plotly_chart(fig)

# Function to resize and normalize input image using OpenCV
def resize_input_image(img, version):
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")

    # Convert PIL Image to numpy array
    img_array = np.array(img)

    # Resize the image using OpenCV
    img_resized = cv2.resize(img_array, (image_shape[1], image_shape[0]))

    # Normalize the image
    my_image = img_resized / 255.0

    # Expand dimensions for compatibility with your model
    my_image = np.expand_dims(my_image, axis=0)

    return my_image

def load_model(model_path):
    return joblib.load(model_path)

# Function to load model and perform prediction
def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """
    model_file_path = "outputs/v1/mildew_detector_model.h5"

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        # If it doesn't exist, create the file or perform any necessary model training here
        # Example: You can create an empty file like this
        with open(model_file_path, 'w') as model_file:
            pass
        print("Mildew detector model file created.")
    else:
        print("Mildew detector model file already exists.")

    model = load_model(f"outputs/{version}/mildew_detector_model.h5")

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {v: k for k, v in {'healthy': 0, 'powdery_mildew': 1}.items()}
    pred_class = target_map[pred_proba > 0.5]
    if pred_class == target_map[0]:
        pred_proba = 0 - pred_proba

    if pred_class.lower() == 'healthy':
        st.write(
            f"The predictive analysis indicates the leaf is "
            f"**healthy**.")
    else:
        st.write(
            f"The predictive analysis indicates the leaf contains "
            f"**powdery mildew**.")

    return pred_proba, pred_class

# Streamlit app main code
def main():
    st.title("Mildew Detection")

    st.info(
        "The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew."
    )

    st.write(
        "You can download a set of healthy and powdery mildew leaves for live prediction. "
        "You can download the images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader('Upload a cherry leaf image. You may select more than one.',
                                     type=['jpeg', 'jpg'], accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:
            st.subheader(f"Cherry Leaf: **{image.name}**")
            img_pil = Image.open(image)

            # Resize and normalize image using OpenCV
            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)

            # Perform prediction and visualization
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            st.subheader(f"**{image.name} Analysis**")
            st.write(f"Result: {pred_class}")

            df_report = df_report.append({"Name": image.name, 'Result': pred_class},
                                         ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.write(df_report)
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
