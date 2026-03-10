# %% 


# ALL vs AML

# The main biological distinction is between two major leukemia families:

# Acute Lymphoblastic Leukemia (ALL)
# cancer of immature lymphoid cells.

# Acute Myeloid Leukemia (AML)
# cancer of immature myeloid cells.


import pandas as pd 
import numpy as np 
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, lasso_path, Lasso, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('golub.csv')
print(df.head())

mapper = {'allB':0,
          'allT':0,
          'aml':1}

df['outcome'] = df['cancer'].map(mapper)


# %%

# 1. Straight linear regression

mse = lambda y,y_hat : np.mean( (y-y_hat)**2 )

y = df['outcome']
X = df.drop(['Samples', 'BM.PB', 'Gender', 'Source', 'tissue.mf', 'cancer','outcome'],axis=1)

model = LinearRegression()
reg = model.fit(X,y)

y_hat = reg.predict(X)
print(f'OLS training MSE: {mse(y,y_hat)}')

residuals = y_hat - y
sns.kdeplot(residuals)
plt.show()
residuals.describe()

sns.scatterplot(x=y,y=y_hat)
plt.show()

# Cross validation of the linear model

kfold = KFold(n_splits=5, shuffle=True, random_state=100) # Create folds
scores = cross_val_score( # Conduct kfcv:
    model,X,y, # Model and data
    cv=kfold, # Folds
    scoring='neg_mean_squared_error' # Loss function
)

mse = -scores

sns.histplot(mse,bins=10).set(title='MSE estiamtes')

print("Fold scores:", mse)
print("Mean score:", np.mean(mse))
print("Median score:", np.median(mse))
print("Std dev:", np.std(mse))


# %%


# 2. Lasso

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

alpha_grid = np.logspace(-4, -2, num=50)
model = LassoCV(cv=10, 
                alphas=alpha_grid,
                random_state=100)
model = model.fit(X_sc, y)

alpha_star = model.alpha_
index_star = np.argmin( np.mean(model.mse_path_,axis=1) )
coefs_star = Lasso(alpha=alpha_star, max_iter=10000).fit(X_sc,y).coef_

# %%

sns.lineplot( x=model.alphas_, y= np.mean(model.mse_path_,axis=1) )
plt.axvline(x=alpha_star, color='green', linestyle='--', 
            linewidth=1.5)
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Cross Validated MSE")
print(f'Optimal cost hyperparameter: {alpha_star}')

# %%

coefs = []
for alpha in model.alphas_: # For each alpha value,
    reg = Lasso(alpha=alpha, max_iter=10_000) # Create a lasso model
    reg = reg.fit(X_sc,y) # Run the regression
    coefs.append(reg.coef_) # Save the slope coefficients
coefs = np.array(coefs) # Cast list of lists to array

plt.figure()
for i in range(coefs.shape[1]):
    plt.plot(model.alphas_, coefs[:, i], label=X.columns[i]) # Switched in poly_names
plt.xscale("log")
plt.xlabel("alpha")
plt.ylabel("Coefficient value")
plt.title("LASSO Coefficient Paths")
plt.axvline(x=alpha_star, color='green', linestyle='--')
plt.show()

# %%

coefs_star = coefs[index_star]
sns.histplot( coefs_star )
plt.show()

nonzero_indices = np.where( coefs_star != 0 )
print('Selected Genes:\n', X.columns[nonzero_indices])
sns.histplot( coefs_star[nonzero_indices] )
plt.show()

print("Nonzero coefficients:", np.sum(coefs_star != 0))
print("Total genes:", len(coefs_star))

for i in nonzero_indices:
    plt.plot(model.alphas_, coefs[:, i], label=X.columns[i])

plt.xscale("log")
plt.axvline(alpha_star, linestyle="--", color="green")
plt.xlabel("alpha")
plt.ylabel("Coefficient value")
plt.title("LASSO Paths (Selected Genes)")
plt.legend(fontsize=6)
plt.show()
