import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import resize_input_image, load_model_and_predict

def main():
    st.title("Mildew Detector")
    st.info("The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.")

    st.write("You can download a set of healthy and powdery mildew leaves for live prediction. You can download the images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).")
    st.write("---")

    images_buffer = st.file_uploader('Upload a cherry leaf image. You may select more than one.', type=['jpeg', 'jpg'], accept_multiple_files=True)

    if images_buffer:
        df_report = pd.DataFrame(columns=['Name', 'Result', 'Confidence'])

        for image in images_buffer:
            st.subheader(f"Cherry Leaf: **{image.name}**")
            img_pil = Image.open(image)
            img_array = np.array(img_pil)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            # Placeholder functions for resizing image and model prediction
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)

            # Convert pred_proba to float if it's not already a float
            pred_proba = float(pred_proba)

            # Ensure that pred_proba is within the valid range
            pred_proba = max(0.0, min(1.0, pred_proba))

           
            
            with col2:
                st.subheader("Prediction")
                st.metric(label="Class", value=pred_class)

                # Displaying the prediction probability as a progress bar
                st.write("Prediction Confidence:")
                st.progress(pred_proba)

                # Displaying the prediction probability as a percentage
                st.metric(label="Confidence", value=f"{pred_proba * 100:.2f}%")

            # Adding the prediction result to the report dataframe
            new_row_df = pd.DataFrame({"Name": [image.name], 'Result': [pred_class], 'Confidence': [f"{pred_proba * 100:.2f}%"]})
            df_report = pd.concat([df_report, new_row_df], ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.dataframe(df_report)
            # Uncomment the below line once you define or import the function `download_dataframe_as_csv`
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
