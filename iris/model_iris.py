# Import dependencies
import pandas as pd
import numpy as np
import joblib

# Load the dataset in a dataframe object and include only four features as mentioned
#url = "https://titanic-buck-ash.s3.us-east-2.amazonaws.com/titanic/train.csv"
#df = pd.read_csv(url)
df = pd.read_csv("Iris.csv")
include = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] # Only four features
df_ = df[include]

# Data Preprocessing               
#categoricals = []
#for col, col_type in df_.dtypes.iteritems():
#     if col_type == 'O':
#          categoricals.append(col)
#     else:
#          df_[col].fillna(0, inplace=True)

#df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
dependent_variable = 'Species'
x = df_[df_.columns.difference([dependent_variable])]
y = df_[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

# Save your model
#from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')
# save the model to disk
#filename = 'finalized_model.sav'
#joblib.dump(lr, filename)
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')
print(lr)

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

