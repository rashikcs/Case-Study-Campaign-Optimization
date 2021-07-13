# Case-Study-Campaign-Optimization
Future Demand: Case study


# Table of contents

Table Of Content for this readme.

- [Case Study](#case-study)
- [Task Solution](#task-solution)
- [Installation](#installation)
- [Project Structure and Usage](#project-structure-and-usage)

# Case Study
Given hourly dataset of different advertisements differentiate the well performing advertisements. 
This solution proposes to processes the data and transforms this problem a supervised classification task to create a baseline model

# Task Solution
  Following approaches were taken to solve this problem

  - **Preliminary data exploration** ->PreProcess.ipynb
  - **Data Cleaning/Processing** ->PreProcess.ipynb
  - **Feature Engineering** ->PreProcess.ipynb
  - **Exploratory Data Analysis** ->EDA.ipynb
  - **Model Training (cross validation and hyperparameter tuning)**->HPO.ipynb
  - **Model Training and Evaluation** ->Modeling/*.ipynb
  - **Gather Insights and further evaluation**-> ->Error Analysis.ipynb

# Installation

  **Requrements**: You need conda 4.9.2 or later to run this project. To use this project, first clone the repo on your device using the command below:
  ```
  git clone https://github.com/rashikcs/Case-Study-Campaign-Optimization.git
  ```
 
  **Installation**: To install required packages and create virtual environment simply run following commands
  ```
  conda env create -f environment.yml
  conda activate campaign_case_study
  ```
  
# Project Structure and Usage
  - **Project Structure**:

        .
        ├── scripts                                # Contain all the necessray python files
        │   ├── preprocess.py                      # Contain all the necessray fuctions for pre processing the given data
        │   ├── utlis.py                           # Contain generic methods
        │   ├── visualization.py                   # Contain methods for visualizations
        │   ├── evaluate.py                        # Contain methods related to evaluate models
        │   ├── model.py                           # Contain methods related to get models used in this project
        ├── plots                                  # Folder containing all the plots genereated
        ├── results                                # Folder containing results of the trained models as csv
        ├── Modeling                               # Folder containing neessary *ipynb files for training model
        ├── trained_models                         # Folder containing saved trained models
        ├── presentations                          # Folder containing the presentation
        ├── data                                   # Expects Folder containing data
        ├── PreProcess.ipynb
        ├── EDA.ipynb
        ├── HPO.ipynb
        ├── Error Analysis.ipynb    
        ├── environments.yml                             
        ├── requirements.txt
        └── README.md
  - **Usage**: Run following notebooks chronologically    

    - PreProcess.ipynb
    - EDA.ipynb
    - HPO.ipynb
    - Modeling/Tree Based Models.ipynb
    - Modeling/Ensemble Model.ipynb
    - Error Analysis.ipynb