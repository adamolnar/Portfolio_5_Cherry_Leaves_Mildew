import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random

def main():
    st.title("Data Visualization")
    page_cells_visualizer_body()

def page_cells_visualizer_body():
    st.write("### Leaves Visualizer")
    st.success(
        " The purpose of this page is to visually illustrate "
        " the disparities between a **healthy** cherry leaf and one afflicted with **powdery mildew**. "
        
    )
    
    version = 'v1'
    if st.checkbox("Difference between average and variability image"):
        avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            "A noticeable visual contrast in the consistency of coloring is evident:"
            "Infected leaves typically exhibit more white blotches on their surface compared to the predominantly greenish,"
            "uniform coloring seen in healthy leaves."
            "However, upon examining the average and variability images, no distinct patterns emerged"
            "that could easily distinguish between infected and healthy leaves."
        )

        st.image(avg_powdery_mildew, caption='Powdery Mildew Leaf - Average and Variability')
        st.image(avg_healthy, caption='Healthy Leaf - Average and Variability')
        st.write("---")

    if st.checkbox("Differences between average powdery mildew and average healthy leaves"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "We recognize a similar trend here, with healthy leaves displaying a clearer, green surface,"
            "while infected leaves exhibit more white coloring."
            "However, comparing both types of leaves remains challenging."
        )
        st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Identification of Differences between Powdery Mildew and Healthy Cherry Leaves Using HSV Color Space"):
        diff_between_avgs = plt.imread("outputs/v1/hsv_comparison.png")

        st.warning(
            "This visualization compares powdery mildew and healthy leaves using the HSV color space, "
            "which breaks down each pixel into its Hue, Saturation, and Value components. "
            "By analyzing these components separately, subtle differences in color characteristics between the two conditions can be identified, "
            "with fungal-affected tissue typically highlighted in bright red color and healthy tissue appearing in brownish hues, "
            "providing insights into the distinct color profiles associated with each leaf condition."
        )
        st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Distinguishing Powdery Mildew from Healthy Leaves Using Greyscale Images"):
        diff_between_avgs = plt.imread("outputs/v1/grayscale_comparison.png")

        st.warning(
            " Grayscale images contain only shades of gray, eliminating color complexity. This simplification can facilitate certain image processing operations and analysis tasks. "
            " In some cases, converting images to grayscale can enhance contrast and reveal details that may be less visible in color images."
            " Contrast between white/powdery mildew marks and grey - representing the healthy part of the leaf, is shown in greyscale: "
            " This approach likely enhances the visibility of the mildew marks against the healthy leaf tissue. "
        )
        st.image(diff_between_avgs, caption='Difference between average images')

    if st.checkbox("Image Montage"): 

        st.warning(
            " The montage aids in visually distinguishing between a healthy leaf and an infected one. "
            " The infected leaf typically displays white, powdery spots or patches on its upper surface."
        )

        st.write("* To refresh the montage, click on **'Create Montage'** button")
        my_data_dir = 'inputs/cherry_leaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')

        # Allow for the selection of multiple labels
        labels_to_display = st.multiselect(label="Select labels", options=labels, default=labels[:2])

        if st.button("Create Montage"):
            with st.spinner("Loading montage..."):
                image_montage(dir_path='inputs/cherry_leaves_dataset/cherry-leaves/validation',
                              labels_to_display=labels_to_display,
                              nrows=4, ncols=3, figsize=(20, 30))  # Adjusted figsize for bigger images
        st.write("---")

def image_montage(dir_path, labels_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    all_labels = os.listdir(dir_path)

    # Check if the selected labels exist in the directory
    valid_labels = [label for label in labels_to_display if label in all_labels]
    if not valid_labels:
        st.write("None of the selected labels exist.")
        return

    images_list = []
    for label in valid_labels:
        label_dir = os.path.join(dir_path, label)
        images_list.extend([os.path.join(label_dir, file) for file in os.listdir(label_dir)])
    
    # Ensure there are enough images to fill the montage
    if nrows * ncols > len(images_list):
        st.write(
            f"Decrease nrows or ncols to create your montage. "
            f"There are only {len(images_list)} images available. "
            f"You requested a montage with {nrows * ncols} spaces."
        )
        return

    img_idx = random.sample(images_list, nrows * ncols)

    # Create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for x in range(nrows * ncols):
        img = imread(img_idx[x])
        img_shape = img.shape
        axes[x // ncols, x % ncols].imshow(img)
        axes[x // ncols, x % ncols].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px", fontsize=10)  # Adjust font size if needed
        axes[x // ncols, x % ncols].axis('off')  # Hide axes ticks
    plt.tight_layout()
    st.pyplot(fig=fig)

if __name__ == "__main__":
    main()
