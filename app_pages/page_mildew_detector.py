import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import base64  # Import base64 module
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import resize_input_image, load_model_and_predict

def download_dataframe_as_csv(dataframe):
    # Create a stream to hold the data
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64

    # Create a download link
    href = f'<a href="data:file/csv;base64,{b64}" download="analysis_report.csv">Download CSV</a>'
    return href

def main():
    st.title("Mildew Detector")
    st.success("The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.")

    st.warning("You can download a set of healthy and powdery mildew leaves for live prediction. You can download the images from [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).")
    st.write("---")

    images_buffer = st.file_uploader('Upload a cherry leaf image. You may select more than one.', type=['jpeg', 'jpg'], accept_multiple_files=True)

    if images_buffer:
        df_report = pd.DataFrame(columns=['Name', 'Result', 'Confidence'])

        for image in images_buffer:
            st.markdown(
                f'<h3 style="background-color: #333; color: white; padding: 8px; border-radius: 8px; margin-bottom:15px;">Cherry Leaf: <strong>{image.name}</strong></h3>',
                unsafe_allow_html=True
            )
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
                st.markdown("<h3 style='background-color: grey; color: white; padding: 8px;'>Prediction</h3>", unsafe_allow_html=True)
                st.metric(label="", value=pred_class)

                # Displaying the prediction probability as a progress bar
                
                st.progress(pred_proba)

                # Displaying the prediction probability as a percentage
                st.metric(label="Confidence", value=f"{pred_proba * 100:.2f}%")
                

            # Adding the prediction result to the report dataframe
            new_row_df = pd.DataFrame({"Name": [image.name], 'Result': [pred_class], 'Confidence': [f"{pred_proba * 100:.2f}%"]})
            df_report = pd.concat([df_report, new_row_df], ignore_index=True)
            st.markdown("---")

        if not df_report.empty:
            st.info("Analysis Report")
            st.dataframe(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
