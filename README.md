 heart_disease_predictor
 🫀 Heart Disease Predictor using Machine Learning

This project builds a machine learning model to predict the presence of heart disease using clinical features. It leverages Python's powerful data science libraries and is optimized for execution in Google Colab.

 🚀 Features

- Uses a clean public heart disease dataset
- Preprocesses data with scaling and label encoding
- Trains a Random Forest Classifier
- Evaluates performance with:
  - Accuracy and classification report
  - Confusion matrix
  - ROC curve
  - Correlation heatmap
  - Target class distribution

 📁 Project Structure
 heart_disease_predictor/
├── heart_disease_predictor.ipynb 
├── README.md
Sample Output
✅ Accuracy: ~82%

📄 Precision, Recall, F1-score shown in classification report

📉 Visualizations: Confusion Matrix, ROC Curve, Heatmap, Count Plots
 📊 Dataset

- 📎 **Source:** [TensorFlow Heart CSV](https://storage.googleapis.com/download.tensorflow.org/data/heart.csv)
- 📈 **Rows:** 303 patients  
- 🔍 **Features:**  
  - age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG, max heart rate, exercise-induced angina, ST depression, slope, number of vessels, thalassemia
- 🎯 **Target:**  
  - `0 = No Heart Disease`, `1 = Heart Disease`

 🛠️ Requirements

Ensure these libraries are installed (or run in Google Colab where they’re pre-installed):

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
