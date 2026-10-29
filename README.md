<H3>ENTER YOUR NAME: Naveen R
<H3>ENTER YOUR REGISTER NO:212225040276
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the CSV file uploaded through the Colab Files sidebar
data = pd.read_csv("Churn_Modelling (2).csv")

# Display initial records
print("First five records:")
display(data.head())

# Check dataset details
print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())
print("\nDataset information:")
data.info()

# Separate input features and output variable
X = data.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
y = data["Exited"]

print("\nInput features:")
display(X.head())

print("\nOutput variable:")
print(y.head())
print("\nChurn counts:")
print(y.value_counts())

# Split data into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Encode categorical columns and normalize numerical columns
categorical_columns = ["Geography", "Gender"]
numerical_columns = [column for column in X.columns if column not in categorical_columns]

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numerical_columns),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# Fit preprocessing only on training data, then transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Display final results
print("\nExperiment 1 completed successfully.")
print("\nDataset shape:", data.shape)
print("Missing values:", data.isnull().sum().sum())
print("Duplicate rows:", data.duplicated().sum())

print("\nProcessed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)

print("\nFinal feature names:")
print(preprocessor.get_feature_names_out())

## OUTPUT:
<img width="1917" height="260" alt="Screenshot 2026-10-29 185229" src="https://github.com/user-attachments/assets/a47d7c58-a08f-4bca-9ce2-741797809d07" />

<img width="555" height="715" alt="Screenshot 2026-10-29 185329" src="https://github.com/user-attachments/assets/da5f329c-bbf4-4a71-b9c5-daf167fd1a13" />

<img width="1007" height="506" alt="Screenshot 2026-10-29 185534" src="https://github.com/user-attachments/assets/09cc4338-7053-487d-ad79-a8bd6c6ae7bd" />


<img width="322" height="52" alt="Screenshot 2026-10-29 185557" src="https://github.com/user-
attachments/assets/ee7f9802-9292-4f9b-96b1-23158d0e2ea5" />

<img width="321" height="107" alt="Screenshot 2026-10-29 185550" src="https://github.com/user-attachments/assets/bc17fdb7-4f37-4c86-a4fa-90001fc21836" />

<img width="545" height="292" alt="Screenshot 2026-10-29 185612" src="https://github.com/user-attachments/assets/5d8474da-33a5-45f5-807e-b085c1c0de8c" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


