import streamlit as st

def main():
    st.title("Project Hypothesis and Validation")

    st.success(
        f"### Hypothesis 1: Image Analysis for Powdery Mildew Identification\n"
        f" The hypothesis is that the presence of powdery mildew on leaves can be accurately identified through image analysis.\n"
        f"\n **Validation Methods:**\n"
        f"  - Average Image Comparison: Powdery mildew-affected leaves tend to be lighter in color compared to healthy leaves.\n"
        f"  - Variability and Average Difference Images: Clear contrast variation around the middle of healthy leaves: This implies that in healthy leaves,\n"
        f"  there are noticeable differences or contrasts in the middle portion compared to the surrounding areas. This difference might be indicative of a characteristic associated with health or vitality.\n"
        f"  - HSV (Hue, Saturation, Value) Color Space Conversion: Leaf images were converted to the HSV color space to analyze color variations and patterns. \n"
        f" characteristics between the two conditions can be identified, with fungal-affected tissue typically highlighted in bright red color and healthy tissue appearing in brownish hues\n"
        f"  - Grayscale Visualization: Typical characteristics of infected leaf include display of particular white marks in contrast with grey - healthy leaf tissue.\n"
        f"  - Image Montage Analysis: Powdery mildew-affected leaves exhibit patches of white coating and discoloration.\n"

    )

    st.success(
        f"### Hypothesis 2: Machine Learning for Cherry Leaf Health Prediction\n"
        f" The hypothesis is that machine learning can predict if a cherry leaf is healthy or contains powdery mildew based on leaf images with at least 97% accuracy..\n"
        f" User-friendly dashboard can be developed to provide instant cherry leaf health assessments based on uploaded images.\n"
        f"\n **Validation Methods:**\n"
        f"  - Machine Learning Model: A machine learning model was trained using a dataset of labeled leaf images to automate the identification process. This model was validated against real-world samples."
        f"  - Development of Dashboard: A user-friendly dashboard was developed to provide instant cherry leaf health assessments based on uploaded images."
    )

    st.warning(
        f"### Conclusions\n"
        f"**The conclusions were drawn based on the observations and methods mentioned above:**\n"
        f"- The image analysis approach has shown promising results in accurately identifying powdery mildew-affected leaves from healthy leaves with at least 97% accuracy..\n"
        f"- Image montage analysis and average image comparison provided valuable insights into the visual differences between healthy and affected leaves.\n"
        f"- Variability and average difference images highlighted contrast variations, aiding in the identification process.\n"
        f"- The machine learning model demonstrated strong potential for automation, and ongoing validation will provide further insights into its effectiveness.\n"
        f"- Further experimentation and validation are ongoing to refine the method and ensure its reliability in real-world scenarios.\n"
        f"- The findings suggest that automated image analysis can be a valuable tool in agricultural disease detection and management."
    )

if __name__ == "__main__":
    main()
