import streamlit as st

# Define your MultiPage class here
class MultiPage:
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        # Create a menu to switch between pages
        selected_page = st.sidebar.radio('Go to', self.pages, format_func=lambda page: page['title'])

        # Display the selected page's title and content
        with st.container():
            st.subheader(selected_page['title'])
            selected_page['function']()