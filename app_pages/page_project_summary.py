import streamlit as st

def main():
    st.title("Project Summary")

    st.success(
        f"**General Information**\n"
        f"* Cherry Leaf Mildew Detection is a project designed to help users analyze and predict whether cherry leaves are healthy or contain powdery mildew, a common fungal disease that affects cherry trees.\n"
        f"* Powdery mildew can significantly impact crop yield and quality.\n"
        f"* Early detection of powdery mildew is crucial for effective disease management.\n"
        f"* According to research, powdery mildew is primarily caused by fungi in the Erysiphaceae family and manifests as a white, powdery substance on the surface of cherry leaves.\n"
        f"* Detecting powdery mildew visually and determining its presence are key goals of this project."
    )

    st.warning(
        f"**Project Dataset**\n"
        f"* The dataset is sourced from [KAGGLE](https://www.kaggle.com/codeinstitute/cherry-leaves).\n"
        f"* The project dataset includes 4208 of images of cherry leaves, both healthy and affected by powdery mildew.\n"
        f"* The images dataset has been split evenly with\n"
        f"  healthy cherry leaves and leaves with powdery Mildew.\n"
        f"* These images serve as the foundation for training a machine learning model to detect the presence of mildew.\n"
        f"* For additional details, please refer to the project's [README file](https://github.com/adamolnar/Portfolio_5_Cherry_Leaves_Mildew).\n"
    )

    st.info(
        f"The project is driven by two primary business requirements:\n"
        f"* 1 - Visual Differentiation: The client seeks to visually differentiate between cherry leaves that are parasitized with powdery mildew and those that are not.\n"
        f"* 2 - Powdery Mildew Detection: The client aims to determine whether a given cherry leaf contains the powdery mildew fungi.\n"
    )   

if __name__ == "__main__":
    main()