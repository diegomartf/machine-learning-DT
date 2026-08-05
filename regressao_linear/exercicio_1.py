#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#%%
arquivo = pd.read_csv('kc_house_data.csv')
arquivo.head()

# %%
X = arquivo[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']]
y = arquivo['price']

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#%%
score = modelo.score(X_test, y_test)
print(f"R² score: {score:.4f}")