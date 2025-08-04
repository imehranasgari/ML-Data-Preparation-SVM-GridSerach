# svm - GridSearch & Cross-Validation for Machine Learning Models

## Problem Statement
The project aims to evaluate and improve the generalization capability of a predictive machine learning model, specifically addressing issues like overfitting and underfitting. It also seeks to find the optimal hyperparameters for the model to achieve the best performance on unseen data.

## Solution Approach
The solution involves:
*   **Cross-Validation**: Employing cross-validation techniques to assess how well the model generalizes to independent datasets and to minimize overfitting and underfitting.
*   **Grid Search**: Utilizing Grid Search to systematically search for the best combination of hyperparameters for the machine learning model.

The process includes:
1.  Importing necessary libraries for data manipulation, visualization, and machine learning.
2.  Loading and inspecting the "Social\_Network\_Ads.csv" dataset.
3.  Splitting the dataset into training and testing sets.
4.  Applying Feature Scaling to the independent variables.
5.  Training a Kernel SVM model on the training set.
6.  Evaluating the model's performance using a Confusion Matrix and calculating accuracy.
7.  Applying k-Fold Cross Validation to get a more robust accuracy measure and standard deviation.
8.  Applying Grid Search to find the optimal hyperparameters and the best model configuration.

## Technologies & Libraries
*   **Python**: Programming language.
*   **NumPy**: For numerical operations.
*   **Pandas**: For data manipulation and analysis.
*   **Matplotlib**: For plotting and visualization.
*   **Seaborn**: For statistical data visualization.
*   **Scikit-learn (sklearn)**: For machine learning functionalities, including:
    *   `train_test_split` for splitting data.
    *   `StandardScaler` for feature scaling.
    *   `SVC` for Support Vector Classification (Kernel SVM).
    *   `confusion_matrix` and `accuracy_score` for model evaluation.
    *   `cross_val_score` for k-Fold Cross Validation.
    *   `GridSearchCV` for hyperparameter tuning.

## Installation & Execution Guide

### Prerequisites
Ensure you have Python installed. You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Dataset
The project uses a dataset named `Social_Network_Ads.csv`. Ensure this file is in the same directory as the Jupyter notebook.

### Execution
1.  Open the `GridSerach & Cross-Validatoin.ipynb` notebook in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, VS Code with Jupyter extension).
2.  Run all cells sequentially to execute the data loading, preprocessing, model training, cross-validation, and grid search steps.

## Key Results / Performance

### Initial Model Performance (Kernel SVM)
*   **Confusion Matrix**:
    ```
    [[64  4]
     [ 3 29]]
    ```
*   **Accuracy Score**: 0.93 (93%)

### K-Fold Cross Validation Performance
*   **Average Accuracy**: 90.33 %
*   **Standard Deviation**: 6.57 %

### Grid Search Results (Optimal Hyperparameters)
*   **Best Accuracy**: 90.67 %
*   **Best Parameters**: `{'C': 0.5, 'gamma': 0.6, 'kernel': 'rbf'}`

## Screenshots / Sample Outputs

### Cross-Validation Concept
![Cross-Validation Image](https://cdn.hackernoon.com/hn-images/1*3XvSvKfde8u89TMwjkz3kg.png)

### Hold-out vs. Cross-validation
![Hold-out vs. Cross-validation Image](https://miro.medium.com/max/875/1*pJ5jQHPfHDyuJa4-7LR11Q.png)

### Different Validation Strategies
![Different Validation Strategies Image](https://www.upgrad.com/blog/wp-content/uploads/2020/02/screenshot-miro.medium.com-2020.02.14-17_27_05.png)

### K-Fold Cross Validation Visual
![K-Fold Cross Validation Visual](https://miro.medium.com/max/1400/1*4G__SV580CxFj78o9yUXuQ.png)

### Grid Search Visual
![Grid Search Visual](https://miro.medium.com/max/1400/0*0SGzQwbkOwSmE13A)

---

# **Author:** mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.