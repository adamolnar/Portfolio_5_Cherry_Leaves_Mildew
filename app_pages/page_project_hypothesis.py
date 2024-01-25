import streamlit as st

def main():
    st.title("Project Hypothesis and Validation")
    st.header("Hypothesis:")
    st.write("We hypothesize that the presence of powdery mildew on leaves can be accurately identified through image analysis.")
    st.header("Validation Methods:")
    st.write("To validate our hypothesis, we have employed the following methods and observations:")
    st.markdown("- **Image Montage Analysis**: Powdery mildew affected leaves exhibit patches of white coating and discoloration.")
    st.markdown("- **Average Image Comparison**: Powdery mildew affected leaves tend to be lighter in color compared to healthy leaves.")
    st.markdown("- **Variability and Average Difference Images**: There is no significant variation around the middle of either leaf, but clear contrast variation is observed around the middle of healthy leaves.")
    st.markdown("- **Texture Analysis**: We have also applied texture analysis techniques to identify unique patterns associated with powdery mildew, such as the fine, powdery texture that distinguishes it from the smooth surface of healthy leaves.")
    st.markdown("- **Machine Learning Model**: We are training a machine learning model using a dataset of labeled leaf images to automate the identification process. This model will be validated against real-world samples.")
     

    st.header("Conclusion:")
    st.write("Based on the observations and methods mentioned above, we believe that our image analysis approach can effectively distinguish powdery mildew-affected leaves from healthy leaves. Further experimentation and validation are ongoing to refine our method.")


if __name__ == "__main__":
    main()