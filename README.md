# Heart-observation---Distributed-ML-for-Marine-Debris-detection

![Project Status: In Progress](https://img.shields.io/badge/status-in%20progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Main Libraries](https://img.shields.io/badge/libraries-Spark%20%7C%20Scikit--learn%20%7C%20Pandas-orange)

This repository contains the work for a university project for the "Distributed Data Analysis and Mining" course. The project focuses on applying distributed computing and machine learning techniques to detect, classify, and analyze floating marine debris from satellite and sensor data.

---

### ⚠ Project Status

*This is an active university project currently in its initial planning phase.* The structure and code will be developed and pushed to this repository starting from October 2025. The pipeline and goals described below represent the planned workflow, which may evolve as the project progresses.

---

## Project Pipeline

This project follows a structured data analysis pipeline as required by the course syllabus.

### 1. Dataset Description

* *Data Source:* The project utilizes the "Floating Marine Debris Data" dataset, curated by Miguel Mendes Duarte. It is publicly available on [this GitHub repository](https://github.com/miguelmendesduarte/Floating-Marine-Debris-Data).
* *Content:* The dataset is composed of sensor measurements simulating data that could be acquired from Earth Observation platforms. It likely includes spectral bands, GPS coordinates, and other sensor readings associated with different classes of marine debris (e.g., plastic, wood, seaweed) and surrounding water.
* *Structure:* The data is provided in CSV format, suitable for analysis with frameworks like Pandas and Spark.

### 2. Project Goal & Research Questions

The primary goal is to build robust machine learning models capable of automatically identifying and classifying marine debris from sensor data, leveraging distributed computing to handle potentially large datasets efficiently.

Our main research questions are:
1.  *Classification Accuracy:* Can we develop a supervised learning model that accurately classifies different types of marine debris (e.g., plastics, natural debris, water) with high precision and recall?
2.  *Feature Importance:* What are the most predictive sensor features (e.g., specific spectral bands, texture features) for distinguishing man-made debris from natural objects?
3.  *Scalability:* How does a distributed computing approach (using Apache Spark or Dask) compare to a single-machine solution in terms of training time and performance as the dataset size increases?

### 3. Data Understanding (Exploratory Data Analysis)

This phase will involve a deep dive into the dataset to uncover patterns, anomalies, and correlations. Planned steps include:
* Calculating descriptive statistics for all numerical features.
* Visualizing data distributions using histograms and density plots.
* Analyzing the correlation between features with a heatmap.
* Visualizing the class distribution to check for imbalances.

### 4. Data Pre-processing

Before model training, the data will be cleaned and transformed. This will include:
* Handling missing values.
* Feature scaling (e.g., Standardization or Normalization) to prepare data for ML algorithms.
* Encoding categorical variables if present.
* Splitting the data into training and testing sets.

### 5. Learning Tasks (Supervised and Unsupervised)

We plan to explore at least two distinct learning tasks:

* *Task 1: Supervised Classification*
    * *Objective:* To classify each data point into a predefined category of marine debris.
    * *Potential Algorithms:* Random Forest, Gradient Boosting (XGBoost, LightGBM), and a simple Multi-Layer Perceptron (MLP). The performance of these models will be rigorously evaluated.

* *Task 2: Unsupervised Clustering*
    * *Objective:* To identify natural groupings within the data without using predefined labels. This could help discover sub-classes of debris or identify anomalous readings.
    * *Potential Algorithms:* K-Means, DBSCAN.

### 6. Draw Conclusions

This final section will summarize the project's findings. We will:
* Present the performance metrics of the trained models.
* Provide clear answers to the initial research questions.
* Discuss the insights gained from the analysis, the limitations of our approach, and potential avenues for future work, such as deploying the model for real-time monitoring.

---

### 🛠 Technology Stack

* *Language:* Python
* *Data Manipulation & Analysis:* Pandas, NumPy
* *Distributed Computing:* Apache Spark (PySpark) or Dask
* *Machine Learning:* Scikit-learn, TensorFlow/Keras (optional)
* *Data Visualization:* Matplotlib, Seaborn
* *Version Control:* Git, GitHub
