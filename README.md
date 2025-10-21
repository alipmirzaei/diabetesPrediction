---

# Diabetes Prediction with Machine Learning: A Genetic Algorithm Approach

This repository contains a comprehensive study on predicting diabetes using various machine learning techniques. The project explores a range of algorithms, including SVM, TwinSVM, Decision Tree, Random Forest, Bagging, and XGBoost.

A key focus of this study is the application of a **Genetic Algorithm (GA)** for feature selection to identify the optimal subset of features for prediction, thereby enhancing model performance and reducing complexity.

---

## 📖 Abstract

This project presents a comparative analysis of multiple machine learning models for diabetes prediction. We explore a range of algorithms to identify the most effective model for this classification task. The study emphasizes the importance of feature selection, employing a Genetic Algorithm (GA) to determine the optimal subset of features. Data preprocessing steps, such as handling class imbalance using SMOTEENN and feature scaling, are also detailed. Performance evaluation is conducted using accuracy, precision, recall, and F1-score. The results demonstrate that **XGBoost** and a **Decision Tree enhanced with GA-based feature selection** achieve the highest accuracy, highlighting the power of both gradient boosting and optimized feature selection.

---

## 🤖 Models & Techniques

This project implements and evaluates the following machine learning models and techniques:

- **Feature Selection:** Genetic Algorithm (GA) using the `deap` library.
- **Classification Models:**
    - Decision Tree (DT)
    - Random Forest (RF)
    - XGBoost
    - Support Vector Machine (SVM) - *from scikit-learn & custom implementation*
    - Twin Support Vector Machine (TwinSVM) - *custom implementation*
    - Bagging Classifier (with SVM as the base estimator)
- **Data Preprocessing:**
    - Handling class imbalance with `SMOTEENN`.
    - One-hot encoding for categorical features.
    - Feature scaling using `StandardScaler`.

---

## 📊 Dataset

The study utilizes the **"Diabetes Prediction Dataset"** which contains various patient attributes, including:
- `age`, `gender`, `bmi`
- `hypertension`, `heart_disease`
- `smoking_history`
- `HbA1c_level`, `blood_glucose_level`

The target variable is `diabetes`, indicating whether a patient has diabetes (1) or not (0).

---

## ✨ Feature Selection with Genetic Algorithm

A Genetic Algorithm was implemented to identify the most significant features for predicting diabetes. The GA was configured as follows:
- **Fitness Function:** The accuracy of a Decision Tree classifier on the test set.
- **Population:** 20 individuals.
- **Generations:** 10.
- **Operators:** Tournament selection, two-point crossover, and bit-flip mutation.

The GA rapidly converged to a high accuracy level, indicating the effectiveness of the feature selection process.

#### Selected Features
The GA identified the following two features as the most relevant for diabetes prediction:
1.  **HbA1c_level**
2.  **blood_glucose_level**

This result underscores the clinical significance of these two metrics in diagnosing diabetes. A final Decision Tree model was trained using only these selected features, achieving performance comparable to the best models trained on all features.

---

## 📈 Results

The performance of each model was evaluated on the test set. The results demonstrate that the **XGBoost** model and the **Decision Tree trained with GA-selected features** achieved the highest accuracy, both around **97.2%**.

### Performance of Different Models

| Model                  | Accuracy | Precision | Recall | F1-Score |
| ---------------------- | -------- | --------- | ------ | -------- |
| SVM (scikit-learn)     | 0.8709   | 0.93      | 0.87   | 0.89     |
| TwinSVM                | 0.8761   | 0.93      | 0.88   | 0.90     |
| Decision Tree          | 0.9141   | 0.94      | 0.91   | 0.92     |
| Random Forest          | 0.9018   | 0.94      | 0.90   | 0.91     |
| Bagging with SVM       | 0.8763   | 0.93      | 0.88   | 0.90     |
| **XGBoost**            | **0.9720**   | **0.97**      | **0.97**   | **0.97**     |
| **GA + Decision Tree** | **0.9719**   | **0.97**      | **0.97**   | **0.97**     |

---

## 🚀 Getting Started

To run this project locally, follow these steps.

### Prerequisites

Ensure you have Python 3 installed. The required libraries are listed in the notebook and can be installed via pip.

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn deap matplotlib seaborn jupyter
```

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Place the dataset:**
    Ensure the `diabetes_prediction_dataset.csv` file is in the root directory.

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook diabetesPredictionwithGA.ipynb
    ```
    Execute the cells in the notebook to preprocess the data, run the GA, and evaluate all the models.
