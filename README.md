# 🚀 Spaceship Titanic Survival Prediction

## Overview
This repository contains a machine learning pipeline to solve the **Spaceship Titanic** Kaggle competition. The goal is to predict whether a passenger was successfully transported to an alternate dimension during the spaceship's collision with a spacetime anomaly. 

This project uses **TensorFlow Decision Forests (TF-DF)** to build a Random Forest classifier, achieving a validation accuracy of **~79.5%**.

## 📊 Dataset
The training dataset consists of **8,693 passengers** and **14 features**, including:
* **Demographics:** `Age`, `HomePlanet`, `Destination`
* **Ship Status:** `CryoSleep`, `Cabin`, `VIP` status
* **Spending Metrics:** `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
* **Target Variable:** `Transported` (Boolean)

*Note: The target variable is perfectly balanced (True: 4378, False: 4315).*

## 🛠️ Data Pipeline & Feature Engineering
Before training the model, the data undergoes several preprocessing steps to clean and extract meaningful signals:
1. **Feature Dropping:** Removed non-predictive identifiers (`PassengerId` and `Name`).
2. **Missing Value Handling:** Missing values for numeric features (`FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `RoomService`) and specific categorical flags (`VIP`, `CryoSleep`) were filled with `0`. TF-DF natively handles remaining categorical missing values.
3. **Type Conversion:** Converted boolean features (`Transported`, `CryoSleep`, `VIP`) to integers for compatibility.
4. **Feature Engineering:**
   * **Cabin Parsing:** Split the `Cabin` string (Format: `Deck/Num/Side`) into three distinct categorical features: `Deck`, `Cabin_num`, and `Side`.
   * **Total Spend:** Consolidated the financial features into a single `TotalSpend` feature representing the sum of all onboard amenity spending.

## 🤖 Modeling
* **Framework:** TensorFlow Decision Forests (`tfdf`)
* **Algorithm:** Random Forest (`tfdf.keras.RandomForestModel`)
* **Hyperparameters:** `num_trees=161`
* **Data Format:** Pandas DataFrames converted to TensorFlow Datasets.

## 📈 Evaluation & Results
The dataset was split into an 80/20 train-validation configuration. 
* **Training Evaluation:** The model's Out-Of-Bag (OOB) score was tracked to monitor accuracy improvements as the number of trees increased to 161.
* **Validation Accuracy:** **79.53%**

## 💻 Technologies Used
* **Python 3**
* **TensorFlow & TensorFlow Decision Forests** (Modeling)
* **Pandas & NumPy** (Data manipulation)
* **Matplotlib & Seaborn** (Data visualization)
* **Scikit-Learn** (Data splitting)
