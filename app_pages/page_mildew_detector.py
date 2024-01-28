import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import resize_input_image, load_model_and_predict, plot_predictions_probabilities

def main():
    st.title("Mildew Detector")
    st.info("The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.")

    st.write("You can download a set of healthy and powdery mildew leaves for live prediction. You can download the images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).")
    st.write("---")

    images_buffer = st.file_uploader('Upload a cherry leaf image. You may select more than one.', type=['jpeg', 'jpg'], accept_multiple_files=True)

    # Initialize df_report here if you want it reset every time the function runs
    df_report = pd.DataFrame(columns=['Name', 'Result'])

    if images_buffer is not None:
        for image in images_buffer:
            st.subheader(f"Cherry Leaf: **{image.name}**")
            img_pil = Image.open(image)
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            st.subheader(f"**{image.name} Analysis**")
            st.write(f"Result: {pred_class}")

            new_row_df = pd.DataFrame({"Name": [image.name], 'Result': [pred_class]})

            df_report = pd.concat([df_report, new_row_df], ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.write(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
