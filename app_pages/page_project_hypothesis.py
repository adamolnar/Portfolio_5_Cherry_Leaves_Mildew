import streamlit as st

def main():
    st.title("Project Hypothesis and Validation")

    st.info(
        f"### Hypothesis 1: Image Analysis for Powdery Mildew Identification\n"
        f"* The hypothesis is that the presence of powdery mildew on leaves can be accurately identified through image analysis.\n"
        f"* Validation Methods:\n"
        f"  - Image Montage Analysis: Powdery mildew-affected leaves exhibit patches of white coating and discoloration.\n"
        f"  - Average Image Comparison: Powdery mildew-affected leaves tend to be lighter in color compared to healthy leaves.\n"
        f"  - Variability and Average Difference Images: There is no significant variation around the middle of either leaf, but clear contrast variation is observed around the middle of healthy leaves.\n"
        f"  - Grayscale Visualization: Grayscale images were generated from the original leaf images to analyze texture and intensity variations.\n"
        f"  - HSV (Hue, Saturation, Value) Color Space Conversion: Leaf images were converted to the HSV color space to analyze color variations and patterns."
    )

    st.info(
        f"### Hypothesis 2: Machine Learning for Cherry Leaf Health Prediction\n"
        f"* The hypothesis is that machine learning can predict if a cherry leaf is healthy or contains powdery mildew based on leaf images.\n"
        f"* Validation Methods:\n"
        f"  - Machine Learning Model: A machine learning model was trained using a dataset of labeled leaf images to automate the identification process. This model was validated against real-world samples."
    )

    st.info(
        f"### Hypothesis 3: Dashboard Development for Instant Health Assessment\n"
        f"* The hypothesis is that a user-friendly dashboard can be developed to provide instant cherry leaf health assessments based on uploaded images.\n"
        f"* Validation Methods:\n"
        f"  - Development of Dashboard: A user-friendly dashboard was developed to provide instant cherry leaf health assessments based on uploaded images."
    )

    st.success(
        f"### Conclusions\n"
        f"**The conclusions were drawn based on the observations and methods mentioned above:**\n"
        f"- The image analysis approach has shown promising results in accurately identifying powdery mildew-affected leaves from healthy leaves.\n"
        f"- Image montage analysis and average image comparison provided valuable insights into the visual differences between healthy and affected leaves.\n"
        f"- Variability and average difference images highlighted contrast variations, aiding in the identification process.\n"
        f"- The machine learning model demonstrated strong potential for automation, and ongoing validation will provide further insights into its effectiveness.\n"
        f"- Further experimentation and validation are ongoing to refine the method and ensure its reliability in real-world scenarios.\n"
        f"- The findings suggest that automated image analysis can be a valuable tool in agricultural disease detection and management."
    )

if __name__ == "__main__":
    main()
