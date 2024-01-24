import streamlit as st

def show_home_page():
    st.title("Welcome to the Cherry Leaf Mildew Detection App.")

    # Create a container with green success background
    with st.beta_container():
        with st.spinner("Loading..."):  # Optional loading spinner
            st.success("This app is designed to help you analyze and predict whether cherry leaves are healthy or contain powdery mildew.")
            st.markdown("## How to Use")
            st.write("Use the navigation menu on the left to explore different sections of the app. You can access the following pages:")
            st.markdown("- **Project Summary**: Learn about the business requirements and objectives.")
            st.markdown("- **Cherry Leaf Visualizer**: Explore visualizations and statistics about cherry leaf data.")
            st.markdown("- **Mildew Detection**: Use machine learning to predict leaf health based on uploaded images.")
            st.markdown("- **Project Hypothesis**: Explore the hypothesis and validation process of this project.")
            st.markdown("- **ML Performance Metrics**: Evaluate the performance of the machine learning model.")
            st.write("Feel free to navigate through the app, upload images, and explore the features provided.")

if __name__ == "__main__":
    show_home_page()
