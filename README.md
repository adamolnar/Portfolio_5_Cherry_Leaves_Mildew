# Mildew Detection in Cherry Leaves Project

## Table of Contents
- [Project Overview](#project-overview)
- [Business Requirements](#business-requirements)
- [Dataset Content](#dataset-content)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Hypothesis Validation Page](#hypothesis-validation-page)
- [Mapping Business Requirements to Data Visualizations and ML Tasks](#mapping-business-requirements-to-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Credits](#credits)


## Project Overview
This project aims to develop a solution for differentiating healthy cherry leaves from those affected by powdery mildew using image analysis. Additionally, it involves building a predictive model to automatically classify cherry leaves as healthy or containing powdery mildew.

## Business Requirements:
The project addresses the following business requirements:
1. **Visual Differentiation**: The client requires a study to visually differentiate healthy cherry leaves from those containing powdery mildew.
2. **Parasite Detection**: The client seeks to predict whether a cherry leaf is healthy or infected with powdery mildew.

### Epics and User Stories
| **Epic**                                        | **User Story**                                                                                                                                                                            |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Epic 1: Data Gathering and Preparation**       | - As a data scientist, I want to source the cherry leaf dataset from a reliable data provider. <br> - As a data analyst, I want to clean the dataset by removing missing values and outliers. <br> - As a data engineer, I want to preprocess the images and extract relevant features. |
| **Epic 2: Data Visualization and Cleaning**       | - As a data analyst, I want to visualize the distribution of healthy and infected cherry leaves. <br> - As a data scientist, I want to clean the dataset by standardizing the pixel values. <br> - As a data engineer, I want to split the dataset into training and testing sets. |
| **Epic 3: Model Training and Optimization**       | - As a machine learning engineer, I want to choose an appropriate machine learning algorithm. <br> - As a data scientist, I want to train the model using the training dataset. <br> - As a model optimizer, I want to tune hyperparameters for better performance. <br> - As a model evaluator, I want to evaluate the model's accuracy and precision. |
| **Epic 4: Dashboard Development**                | - As a dashboard developer, I want to design a user-friendly interface. <br> - As a UI designer, I want to create a page for visualizing the dataset summary and client requirements. <br> - As a UI developer, I want to design a page for displaying findings related to cherry leaf differentiation. <br> - As a UI designer, I want to create a page for live prediction of cherry leaf health. <br> - As a UI developer, I want to implement a file uploader widget for image uploads. <br> - As a UI designer, I want to design a page for project hypothesis and validation. <br> - As a UI developer, I want to create a technical page for displaying model performance. |
| **Epic 5: Dashboard Deployment**                | - As a deployment specialist, I want to deploy the dashboard on a web server. <br> - As a web administrator, I want to ensure the dashboard is accessible to authorized users. <br> - As a maintenance team member, I want to monitor and update the dashboard as needed. |

## Dataset Content
**Dataset**: [Cherry Leaf Dataset on Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

The dataset for the Mildew Detection in Cherry Leaves project consists of a collection of cherry leaf images, each labeled to indicate whether the leaf is healthy or affected by powdery mildew. Here are the key details regarding the dataset:

- **Number of Samples**: The dataset comprises 4,208 curated photographs featuring individual cherry leaves set against a neutral backdrop. 

- **Data Format**: The images are typically in common image formats such as JPEG or PNG.

- **Labeling**: Each image in the dataset is labeled with one of two classes:
  - **Healthy**: Cherry leaves that are free from powdery mildew.
  - **Powdery Mildew**: Cherry leaves that show signs of powdery mildew infection.

- **Data Variability**: The dataset includes images of cherry leaves with varying degrees of powdery mildew infection. This variability is essential for training a robust model.

- **Data Collection**: The images were gathered and categorized by Farmy & Foods Company, operating under a confidential non-disclosure agreement (NDA).

- **Data Distribution**: The dataset is divided into training, validation, and test sets for machine learning purposes.

- **Additional Information**: The dataset originally consists of images sized at 256 pixels × 256 pixels. However, for practicality and to meet project requirements, these images have been reshaped to dimensions like 50 × 50. This resizing is essential for managing the model's file size and ensuring that it aligns with the project's specifications effectively.

### Example Image Samples

 **Healthy Leaves** 

![Healthy Cherry Leaf 1](static/images/healthy_leaf.jpg) ![Healthy Cherry Leaf 2](static/images/healthy_leaf_2.jpg) ![Healthy Cherry Leaf 3](static/images/healthy_leaf_3.jpg)

**Powdery Mildew Infected Leaves**

![Powdery Mildew Cherry Leaf 1](static/images/powdery_mildew_1.jpg) ![Powdery Mildew Cherry Leaf 2](static/images/powdery_mildew_2.jpg) ![Powdery Mildew Cherry Leaf 3](static/images/powdery_mildew_3.jpg)




## Project Structure

- `.devcontainers/`: Configuration files for development containers.
- `app_pages/`: Contains different pages of the web application.
- `input/`: Input data and resources.
- `jupyter_notebooks/`: Jupyter notebooks for data exploration and model development.
- `outputs/`: Generates files as part of its operation, such as reports, logs, or data exports
- `src/`: Python source code for the project.
- `static/`: Contains static assets like images and stylesheets.
- `app.py`: Main application script.
- `README.md`: Project documentation.
- `requirements.txt`: Lists project dependencies.


## Installation

To run the project locally, follow these steps:


## Hypothesis Validation Page:
- Hypothesis
  - The hypothesis is that the presence of powdery mildew on leaves can be accurately identified through image analysis.

- Validation Methods
  - To validate the hypothesis, the project team employed the following methods and observations:
    - Image Montage Analysis: Powdery mildew-affected leaves exhibit patches of white coating and discoloration.
    - Average Image Comparison: Powdery mildew-affected leaves tend to be lighter in color compared to healthy leaves.
    - Variability and Average Difference Images: There is no significant variation around the middle of either leaf, but clear contrast variation is observed around the middle of healthy leaves.
    - Machine Learning Model: A machine learning model was trained using a dataset of labeled leaf images to automate the identification process. This model was validated against real-world samples.
 
- Conclusions
  - Based on the observations and methods mentioned above, the project team drew the following conclusions:
    - The image analysis approach has shown promising results in accurately identifying powdery mildew-affected leaves from healthy leaves.
    - Image montage analysis and average image comparison provided valuable insights into the visual differences between healthy and affected leaves.
    - Variability and average difference images highlighted contrast variations, aiding in the identification process.
    - The machine learning model demonstrated strong potential for automation, and ongoing validation will provide further insights into its effectiveness.
    - Further experimentation and validation are ongoing to refine the method and ensure its reliability in real-world scenarios.
    - The findings suggest that automated image analysis can be a valuable tool in agricultural disease detection and management.

## Mapping Business Requirements to Data Visualizations and ML Tasks
- ...


## ML Business Case
- ...


## Dashboard Design
- ...



## Unfixes Bugs
- ...


## Deploymnet 
- ...


## Main Data Analysis and Machine Learning Libraries
- ...


## Credits

### Project Inspiration
- The idea for this project was inspired by [Mildew Detection in Cherry Leaves](https://learn.codeinstitute.net/courses/course-v1:CodeInstitute+PA_PAGPPF+2021_Q4/courseware/bde016cdbd184cdeafd341a73807e138/bd2104eb84de4e48a9df6f685773cbf2/).

### Project Template
- The project base template is sourced from the [milestone-project-mildew-detection-in-cherry-leaves](https://github.com/Code-Institute-Solutions/milestone-project-mildew-detection-in-cherry-leaves) Git repository.

### Dataset Source
- The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

### Project Content Reference
- The project content and structure were influenced by the Code Institute Walkthrough Project [Malaria Detector](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/07a3964f7a72407ea3e073542a2955bd/29ae4b4c67ed45a8a97bb9f4dcfa714b/).

### Favicon Image
- The project favicon image was sourced from [Freepik](https://www.freepik.com/icon/leaf_892917).
