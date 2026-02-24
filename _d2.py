# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./data/CardiacPatientData.csv')
print(df.head())

features = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'SpO2', 'Age', 'GCS', ]
X = (df.loc[:,features])
y = df['Outcome']

# %%

expander = PolynomialFeatures(degree=3,include_bias=False,interaction_only=True) # Create the expander
X = expander.fit_transform(X) 
X_names = expander.get_feature_names_out() 

# %%

model = LinearRegression()

kfold = KFold(n_splits=10, shuffle=True, random_state=100) # Create folds

scores = cross_val_score( # Conduct kfcv:
    model,X,y, # Model and data
    cv=kfold, # Folds
    scoring="neg_mean_squared_error" # Loss function
)
mse = -scores

sns.histplot(mse,bins=10).set(title='MSE estiamtes')

print("Fold scores:", mse)
print("Mean score:", np.mean(mse))
print("Median score:", np.median(mse))
print("Std dev:", np.std(mse))

# %%

mse = []
for train, test in kfold.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    mse.append(mean_squared_error(y_test, y_hat))
mse = np.array(mse)

print("Fold scores:", mse)
print("Mean score:", np.mean(mse))
print("Median score:", np.median(mse))
print("Std dev:", np.std(mse))

# %%
