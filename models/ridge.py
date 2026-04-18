import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# load data
df = pd.read_csv("Algerian_forest_fires_dataset_cleaned.csv")

df.drop(["Unnamed: 0","day","month","year"],axis=1,inplace=True)
df['Classes'] = np.where(df['Classes'].str.contains("not fire"),0,1)

X = df.drop(["FWI"],axis=1)
y = df["FWI"]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model = Ridge()
model.fit(x_train_scaled,y_train)

print("Model trained. Coef:", model.coef_)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR,"ridgeModel.pkl"),"wb") as f:
    pickle.dump(model,f)

with open(os.path.join(BASE_DIR,"StandardScaler.pkl"),"wb") as f:
    pickle.dump(scaler,f)
