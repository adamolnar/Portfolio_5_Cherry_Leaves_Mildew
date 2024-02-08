# Podery Mildew Detection in Cherry Leaves Project

## Table of Contents
- [Project Overview](#project-overview)
- [CRISP-DM](#crisp-dm)
- [Business Requirements](#business-requirements)
- [User Stories](#user-stories)
- [Dataset Content](#dataset-content)
- [Project Structure](#project-structure)
- [Hypothesis Validation Page](#hypothesis-validation-page)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Platforms](#platforms)
- [Languages](#languages)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Credits](#credits)


## Project Overview
This project aims to develop a solution for differentiating healthy cherry leaves from those affected by powdery mildew using image analysis. Additionally, it involves building a predictive model to automatically classify cherry leaves as healthy or containing powdery mildew.

## CRISP-DM 
**Cross-Industry Standard Process for Data Mining**
is a widely used methodology for data mining and machine learning projects. It provides a structured approach to guide teams through the entire data mining process, from understanding business objectives to deploying predictive models.

CRISP-DM is utilized in various industries and domains where data mining and machine learning techniques are applied. It helps organizations and data science teams to:
1. Understand business objectives and requirements.
2. Explore and understand the data available for analysis.
3. Prepare and preprocess data for modeling.
4. Build, evaluate, and fine-tune predictive models.
5. Deploy models into operational systems.
6. Monitor model performance and maintain them over time.

By following the CRISP-DM methodology, organizations can effectively manage and execute data mining projects, leading to better insights and actionable results from their data.

![CRISP-DM Icon](https://upload.wikimedia.org/wikipedia/commons/b/b9/CRISP-DM_Process_Diagram.png)


## Business Requirements:
The cherry plantation crop from Farmy & Foods faces a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is to verify if a given cherry tree contains powdery mildew manually. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and demonstrating visually if the leaf tree is healthy or has powdery mildew. If it has powdery mildew, the employee applies a specific compound to kill the fungus. The time spent using this compound is 1 minute. The company has thousands of cherry trees on multiple farms nationwide. As a result, this manual process could be more scalable due to the time spent in the manual process inspection.

To save time, the IT team suggested an ML system that can detect instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests. If this initiative is successful, there is a realistic chance to replicate this project in all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

The project addresses the following business requirements:
1. **Visual Differentiation**: The client requires a study to visually differentiate healthy cherry leaves from those containing powdery mildew.
2. **Powdery Mildew Detection**: The client seeks to predict whether a cherry leaf is healthy or infected with powdery mildew.
2. **Daschboard**: The client needs a dashboard that meets the above requirements.

## User Stories

| **User Story**                                                                                                                                                                       | **Business Requirement** |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| As a client, I can navigate easily around an interactive dashboard so that I can view and understand the data presented.                                                                  | Visual Differentiation   |
| As a client, I can view an image montage of either healthy or powdery mildew-affected cherry leaves so that I can visually differentiate them.                                        | Visual Differentiation   |
| As a client, I can view and toggle visual graphs of average images (and average image difference) and image variabilities for both healthy and powdery mildew-affected cherry leaves so that I can observe the difference and understand the visual markers that indicate leaf quality better. | Visual Differentiation   |
| As a client, I can access and use a machine learning model so that I can obtain a class prediction on a cherry leaf image provided.                                                      | Powdery Mildew Disease Detection |
| As a client, I can provide new raw data on a cherry leaf and clean it so that I can run the provided model on it.                                                                          | Powdery Mildew Disease Detection |
| As a client, I can feed cleaned data to the dashboard to allow the model to predict it so that I can instantly discover whether a given cherry leaf is healthy or affected by powdery mildew.                                                    | Powdery Mildew Disease Detection |
| As a client, I can save model predictions in a timestamped CSV file so that I can keep an account of the predictions that have been made.                                                   | Powdery Mildew Disease Detection |
| As a client, I can view an explanation of the project's hypotheses so that I can understand the assumptions behind the machine learning model and its predictions.                     | Dashboard Development    |
| As a client, I can view a performance evaluation of the machine learning model so that I can assess its accuracy and effectiveness.                                                        | Dashboard Development    |
| As a client, I can access pages containing the findings from the project's conventional data analysis so that I can gain additional insights into the data and its patterns.         | Dashboard Development    |



## Dataset Content
**Dataset**: [Cherry Leaf Dataset on Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves).

The dataset for the Mildew Detection in Cherry Leaves project consists of a collection of cherry leaf images, each labeled to indicate whether the leaf is healthy or affected by powdery mildew. Here are the key details regarding the dataset:

- **Number of Samples**: The dataset comprises 4208 curated photographs featuring individual cherry leaves set against a neutral backdrop. 

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



## Hypothesis Validation Page:
- Hypothesis
  - The hypothesis is that the presence of powdery mildew on leaves can be accurately identified through image analysis.
  - Machine learning can predict if a cherry leaf is healthy or contains powdery mildew based on leaf images.
  - A user-friendly dashboard can be developed to provide instant cherry leaf health assessments based on uploaded images.

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


## Dashboard Design
- ...

## Manual Testing
- ...

## Unfixes Bugs
- ...


## Deploymnet 
#### Setup Workspace

1. **Repository Creation:**
   - Click on ["Use This Template"](https://github.com/Code-Institute-Solutions/milestone-project-mildew-detection-in-cherry-leaves) button.
   - Select "Create a new repository".
   - Add a repository name and brief description.
   - Click "Create Repository" to create your repository.

2. **Heroku Deployment:**
   - Go to your Heroku account page.
   - Choose "CREATE NEW APP," give it a unique name, and select a geographical region.
   - Add the `heroku/python` buildpack from the Settings tab.
   - From the Deploy tab, choose GitHub as the deployment method, connect to GitHub, and select the project's repository.
   - Select the branch to deploy, then click "Deploy Branch."
   - Click to "Enable Automatic Deploys" or choose "Deploy Branch" from the Manual Deploy section.
   - Wait for the logs to run while the dependencies are installed and the app is being built.
   - The mock terminal is then ready and accessible from a link similar to `https://your-projects-name.herokuapp.com/`.
   - If the slug size is too large, add large files not required for the app to the `.slugignore` file.


3. **Forking the GitHub Project:**
   - To create a copy of the GitHub repository:
     - Navigate to the repository page and click the "Fork" button. This creates a copy on your GitHub account.

4. **Making a Local Clone:**
   - On the repository page, click on the "Code" button.
   - Copy the HTTPS URL to clone the repository.
   - Open your IDE and change the working directory to the desired location.
   - Type `git clone` in the terminal and paste the URL to clone the repository.

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


## Platforms
- **Heroku**: Deployment platform for the project.
- **Jupyter Notebook**: Used for code editing.
- **Kaggle**: Source for downloading datasets.
- **GitHub**: Repository for storing project code.
- **Gitpod**: Writing and managing code, committing to GitHub, and pushing to GitHub Pages.

## Languages
- **Python**
- **Markdown** 

## Main Data Analysis and Machine Learning Libraries
1. ![tensorflow](https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/120px-TensorFlowLogo.svg.png)
   - TensorFlow is an open-source library for machine learning and deep learning tasks. It provides tools for building and training neural networks.

2. ![numpy](https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/120px-NumPy_logo_2020.svg.png)
   - NumPy is a library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.

3. ![scikit-learn](https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/120px-Scikit_learn_logo_small.svg.png)
   - Scikit-learn is a machine learning library that provides simple and efficient tools for data mining and data analysis. It includes various algorithms for classification, regression, clustering, dimensionality reduction, and model selection.

4. ![streamlit](https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png)
   - Streamlit is a library for creating interactive web applications with Python. It allows developers to build data-driven apps quickly and easily by writing simple Python scripts.

5. ![pandas](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/120px-Pandas_logo.svg.png)
   - Pandas is a library for data manipulation and analysis in Python. It provides data structures like DataFrame and Series, along with functions to clean, filter, and transform data.

6. ![matplotlib](https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Created_with_Matplotlib-logo.svg/120px-Created_with_Matplotlib-logo.svg.png)
   - Matplotlib is a plotting library for Python. It allows developers to create static, animated, and interactive visualizations in Python.

7. ![keras](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/120px-Keras_logo.svg.png)
   - Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, Theano, or CNTK. It simplifies the process of building deep learning models by providing a simple and consistent interface.

8. ![plotly](https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Plotly-logo-01-square.png/120px-Plotly-logo-01-square.png)
   - Plotly is a graphing library for Python that makes interactive, publication-quality graphs online. It allows users to create interactive plots, dashboards, and web applications.

9. ![seaborn](https://seaborn.pydata.org/_static/logo-wide-lightbg.svg)
   - Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.


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
