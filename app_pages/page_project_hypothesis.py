import streamlit as st

def main():
    st.title("Project Hypothesis and Validation")

    st.info(
        f"### Hypothesis\n"
        f"* The hypothesis is that the presence of powdery mildew on leaves can be accurately identified through image analysis.\n"
    )

    st.info(
        f"### Validation Methods\n"
        f"**To validate the hypothesis, the project team employed the following methods and observations:**\n"
        f"- **Image Montage Analysis**: Powdery mildew-affected leaves exhibit patches of white coating and discoloration.\n"
        f"- **Average Image Comparison**: Powdery mildew-affected leaves tend to be lighter in color compared to healthy leaves.\n"
        f"- **Variability and Average Difference Images**: There is no significant variation around the middle of either leaf, but clear contrast variation is observed around the middle of healthy leaves.\n"
        f"- **Machine Learning Model**: A machine learning model was trained using a dataset of labeled leaf images to automate the identification process. This model was validated against real-world samples."
    )

    st.success(
        f"### Conclusions\n"
        f"**Based on the observations and methods mentioned above, the project team drew the following conclusions:**\n"
        f"- The image analysis approach has shown promising results in accurately identifying powdery mildew-affected leaves from healthy leaves.\n"
        f"- Image montage analysis and average image comparison provided valuable insights into the visual differences between healthy and affected leaves.\n"
        f"- Variability and average difference images highlighted contrast variations, aiding in the identification process.\n"
        f"- The machine learning model demonstrated strong potential for automation, and ongoing validation will provide further insights into its effectiveness.\n"
        f"- Further experimentation and validation are ongoing to refine the method and ensure its reliability in real-world scenarios.\n"
        f"- The findings suggest that automated image analysis can be a valuable tool in agricultural disease detection and management."
    )

if __name__ == "__main__":
    main()
