

# Predictive Maintenance for Factory Equipment

## 1. Project Abstract

This project addresses the challenges of ensuring uninterrupted operation and minimizing downtime in industrial settings by implementing predictive maintenance for factory equipment. By leveraging advanced analytics, sensor data, and machine learning algorithms, the project aims to predict equipment failures before they occur. This proactive approach allows for timely maintenance interventions, optimizes overall factory efficiency, and reduces costs associated with unexpected breakdowns.

## 2. Features

* **Data Analysis & Visualization:** Analyzes sensor data distributions and correlates features with failure types .
* **Machine Learning Models:** Implements multiple algorithms to predict machine health, including:
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Convolutional Neural Networks (CNN) 




* 
**Failure Prediction:** Predicts specific failure types (e.g., Power Failure, Tool Wear Failure, Overstrain Failure).


* 
**Remaining Useful Life (RUL) Estimation:** Calculates and suggests the available life of the machinery before maintenance is required.



## 3. System Requirements

### Hardware

* 
**Processor:** i3 or above.


* 
**RAM:** 4 GB.


* 
**Hard Disk:** 40 GB.



### Software

* 
**Operating System:** Windows 8 or above.


* 
**Coding Language:** Python.


* 
**IDE/Environment:** Jupyter Notebook.



## 4. Tech Stack & Libraries

The project utilizes the following Python libraries for data processing, visualization, and modeling:

* **TensorFlow / Keras:** For deep learning (CNN) models.
* **NumPy:** For high-performance multi-dimensional array processing.
* **Pandas:** For data manipulation and analysis.
* **Matplotlib / Seaborn:** For plotting graphs and data visualization.
* **Scikit-learn:** For implementing SVM, Decision Tree, Random Forest, and KNN algorithms.

## 5. Dataset

The project uses the **Predictive Maintenance Sensor Dataset** from Kaggle.

* 
**Source:** [Kaggle - Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification).


* 
**Description:** The dataset includes features such as Air Temperature, Process Temperature, Rotational Speed, Torque, and Tool Wear.


* 
**Target:** The `Failure_Type` column is used as the class label for predictions.



## 6. Installation & Setup

### Step 1: Install Python

Ensure Python (version 3.7.4 or newer) is installed on your system.

1. Download the installer from [python.org](https://www.python.org).


2. Run the installer and ensure you check the box **"Add Python 3.7 to PATH"** before clicking "Install Now".


3. Verify installation by opening a command prompt (`cmd`) and typing `python -V`.



### Step 2: Install Required Libraries

Open your command prompt or terminal and install the necessary packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras jupyter

```

### Step 3: Set Up the Project Directory

1. Create a project folder.
2. Download the dataset (`predictive_maintenance.csv`) from the link provided in Section 5.
3. Place the dataset inside a folder named `Dataset` within your project directory.



## 7. How to Run the Project

The project is coded using **Jupyter Notebook**. Follow these steps to execute it:

1. **Launch Jupyter Notebook:**
Open your command prompt/terminal, navigate to your project directory, and type:
```bash
jupyter notebook

```


2. **Open the Notebook File:**
Click on the `.ipynb` file (e.g., `Untitled.ipynb` or the project filename) in the browser interface.
3. **Execute the Cells:**
Run the code blocks sequentially to perform the following operations:
* 
**Import Packages:** Loads necessary libraries (Pandas, Numpy, Sklearn, etc.).


* 
**Load Data:** Reads the `predictive_maintenance.csv` file.


* 
**Data Analysis:** Displays data descriptions, missing value counts, and visualization graphs (histograms, pie charts) .


* 
**Preprocessing:** Performs Label Encoding (converting text to numbers), data shuffling, and normalization (MinMaxScaler) .


* 
**Data Splitting:** Splits the data into 80% training and 20% testing sets.


* 
**Model Training:** Trains SVM, Decision Tree, Random Forest, KNN, and CNN models .


* 
**Evaluation:** Calculates Accuracy, Precision, Recall, and F-Score for each algorithm and plots Confusion Matrices and ROC Curves.


* 
**Prediction:** Runs the prediction logic on test data to identify failure types and calculate "Available Life Maintenance" percentages .





## 8. Model Performance

The project evaluated several algorithms, yielding the following accuracy results:

* 
**CNN (Deep Learning):** ~97% (Highest Accuracy) 


* 
**Random Forest:** ~92% 


* 
**SVM:** ~92% 


* 
**Decision Tree:** ~88% 


* 
**KNN:** ~82% 



## 9. Conclusion

By employing these machine learning models, specifically the CNN algorithm, the system effectively predicts machinery failure with high accuracy. This enables the scheduling of maintenance based on the "Available Life" metric, ensuring continuous production and preventing costly downtime.
