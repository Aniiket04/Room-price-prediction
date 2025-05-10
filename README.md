# Room Price prediction project

## Project Overview 

**Project Title**: Room price prediction**

The goal of this machine learning project is to develop a predictive model that can estimate the price of a room based on two main features: locality and area (in square feet). The project involves using historical data of room prices, the corresponding area, and locality to train the model using Linear Regression.

## Objectives
To predict house prices based on the town (categorical variable) and other numerical features (e.g., area), using Linear Regression.

## Project Structure

### 1. Importing Libraries
The notebook begins by importing essential Python libraries, including:
pandas for data manipulation
numpy for numerical operations
matplotlib.pyplot or seaborn for data visualization
sklearn (from scikit-learn) for machine learning tools
```python
import pandas as pd
import numpy as np
from sklearn import linear_model
```

### 2. Loading the Dataset
The given dataset is loaded using pandas.read_csv(). This dataset contains data about towns with features like:
area, location (input features)
price (target variable)
```python
df=pd.read_csv("Data1.csv.csv")
df
```

### 3. Data processing
3.1 Encoding Categorical Variables:
Converts text labels (e.g. “town”) into numeric dummy bits.
```python
dummies=pd.get_dummies(df.town)
dummies
```
3.2 Combine two DataFrames: df and dummies, side by side (i.e., column-wise).
```python
merged=pd.concat([df,dummies,],axis='columns')
merged
```
axis='columns' (or axis=1) tells pandas to concatenate columns, not rows

3.3 Drop two columns from the merged DataFrame:
```python
final=merged.drop(["town","west windsor"],axis="columns")
final
```
Why drop one of the dummy column?
This is part of avoiding the dummy variable trap — a issue in regression where:
If all dummy variables are included, they are linearly dependent , which breaks assumptions in linear regression.
So we drop one dummy column (like "west windsor") to serve as the baseline category.
This final DataFrame is now ready to be used for training a regression model.

3.4 Import the LinearRegression class from scikit-learn a popular ML library
```python
from sklearn.linear\_model import LinearRegression
model=LinearRegression()
```

### 4. Model Training
Training a Linear Regression model using sklearn.linear_model.LinearRegression
Fitting the model on the training data
```python
x=final.drop('price',axis='columns')
x
y=final.price
y
model.fit(x,y)
```

### 5. Model Evaluation
Predictions on the held-out dataset.
```python
model.predict([[2900,0,1]])
model.score(x,y)
```
The model.score() function evaluates the performance of a trained model.

## Conclusion
This project successfully demonstrates how to build a linear regression model to predict house prices based on the town (location) and potentially other features such as area.The model was able to fit the dataset well, indicating that location has a significant impact on housing prices. 

## Author - Aniket Pal
This project is part of my portfolio, showcasing the machine learning skills essential for data science roles.

-**LinkedIn**: [ www.linkedin.com/in/aniket-pal-098690204 ]
-**Email**: [ aniketspal04@gmail.com ]









