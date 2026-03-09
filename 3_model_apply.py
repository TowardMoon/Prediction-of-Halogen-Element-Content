import pandas as pd
import os
import joblib
import numpy as np

df1 = pd.read_excel("..\\mice_data.xlsx")
X_all = df1[[]]#Features selected during modeling

# load the trained model
model = os.path.join("..\\rf_model_.pkl")
if not os.path.exists(model):
    print("The model file does not exist, please check the path.")
else:
    model_xgb = joblib.load(model)

result_xgb = model_xgb.predict(X_all)
np.array(result_xgb,dtype=float)
print(result_xgb)

# Insert the predicted contents into the input dataset
df1.insert(df1.shape[1], 'P_', result_xgb)

# Output results
df1.to_excel("..\\result_.xlsx")