import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df=pd.read_excel("..\\smote_data.xlsx")

split1=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=11)

for train_val_index,test_index in split1.split(df,df["Label"]):
    df_train_val=df.loc[train_val_index]
    df_test=df.loc[test_index]

xgb=XGBRegressor()


X=df[['','','']]#Features selected through correlation
y=df['']#Target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

param_grid = {
    'base_score': [2, 3],  # mean
    'booster': ['gbtree', 'dart'],
    'max_depth': [3, 7, 10],  # 3-10
    'min_child_weight': [1, 3, 6],  # 1-6
    'n_estimators': [100, 500, 1000],# 10-1000
    'subsample': [0.5, 0.8, 1],  # 0.5-1
    'gamma': [0, 0.3, 0.5],  # 0-0.5
    'learning_rate': [0.01, 0.15, 0.3],  # 0.01-0.3
}

grid_search=GridSearchCV(estimator=xgb,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
grid_search.fit(X_train,y_train)

print("Best parameters found: ",grid_search.best_params_)
print("Best score found: ",grid_search.best_score_)


#Set the parameters of the XGBoost Regressor model
xgb=XGBRegressor()
model_xgb=XGBRegressor(base_score=2,booster='gbtree',max_depth=3,min_child_weight=6,n_estimators= 1000,subsample=0.5,gamma=0,learning_rate=0.3)

model_xgb.fit(X_train,y_train)

joblib.dump(model_xgb,"..\\xgb_model_.pkl")

y_train_predict=model_xgb.predict(X_train)
y_test_predict=model_xgb.predict(X_test)
train_rmse=np.around(np.sqrt(mean_squared_error(y_train,y_train_predict)),decimals=3)
test_rmse=np.around(np.sqrt(mean_squared_error(y_test,y_test_predict)),decimals=3)
train_r2=np.around(r2_score(y_train,y_train_predict),decimals=3)
test_r2 = np.around(r2_score(y_test,y_test_predict),decimals=3)
print("The RMSE on training set is {}".format(train_rmse))
print("The RMSE on test set is {}".format(test_rmse))
print("R2 score on training set is {}".format(train_r2))
print("R2 score on test set is {}".format(test_r2))