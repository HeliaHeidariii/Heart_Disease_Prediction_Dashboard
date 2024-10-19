# Heart_Disease_Prediction_Dashboard
 This Python project implements a graphical user interface (GUI) to predict heart disease based on patient data using machine learning algorithms like KNN, SVM, Decision Tree, Random Forest, and Logistic Regression. The predictions, along with evaluation metrics such as accuracy, recall, precision, and F1 score, are displayed in a simple interface built using Tkinter.
 
##Heart Disease Prediction Dashboard
This project is a Python-based application designed to predict heart disease using various machine learning algorithms. It provides a simple graphical user interface (GUI) built with Tkinter to input patient data and receive predictions. The project evaluates and compares five different machine learning models: KNN, SVM, Decision Tree, Random Forest, and Logistic Regression.

##Features
User-friendly interface: A simple GUI allows users to enter patient data such as age, sex, cholesterol levels, and more.
Machine learning models: Supports five algorithms: KNN, SVM, Decision Tree, Random Forest, and Logistic Regression.
Performance metrics: Displays model evaluation metrics like accuracy, recall, precision, and F1 score for each prediction.
Prediction table: Results are shown in a table with all evaluation scores for easy comparison.

##Technologies Used
Tkinter: For building the graphical user interface.
scikit-learn: Machine learning algorithms, data preprocessing, and evaluation metrics.
pandas: Data manipulation and loading from CSV files.
numpy: Handling numerical operations and input scaling.

##How to Use
Install the required dependencies:

bash
pip install scikit-learn pandas numpy
Prepare the dataset:

Make sure your dataset is located at the specified path or modify the script to load it correctly.
The dataset should contain features like age, sex, cholesterol levels, and other health-related parameters.
Run the script:

bash
python main_true_heart6.py
Enter the patient data in the provided fields and click one of the buttons (KNN, SVM, etc.) to receive predictions.
