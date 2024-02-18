import streamlit as st
from app_pages.multipage import MultiPage
from app_pages.show_home_page import show_home_page
from app_pages.page_leafs_visualizer import main as leafs_visualizer
from app_pages.page_mildew_detector import main as mildew_detector
from app_pages.page_ml_performance import main as ml_performance
from app_pages.page_project_hypothesis import main as project_hypothesis
from app_pages.page_project_summary import main as project_summary

# Set web browser title and favicon as the first Streamlit command
st.set_page_config(
    page_title="Cherry Leaf Mildew Detector",
    page_icon="static/cherry-leaf.png"  # You can use emoji as the favicon, or provide the path to your favicon.ico file
)

# Define app pages
app_pages = {
    "Home": show_home_page,
    "Project Summary": project_summary,
    "Cherry Leaf Visualizer": leafs_visualizer,
    "Mildew Detection": mildew_detector,
    "Project Hypothesis": project_hypothesis,
    "ML Performance Metrics": ml_performance,
}

# Create a MultiPage instance with your app name
multi_page_app = MultiPage(app_name="Cherry Leaf Mildew Detector")

# Add pages to the MultiPage instance
for page_name, page_function in app_pages.items():
    multi_page_app.add_page(page_name, page_function)

# Run the MultiPage app within the if __name__ block
if __name__ == "__main__":
    multi_page_app.run()
