import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib


data=pd.read_excel('..\\smote_data.xlsx')
X=data[['','','']]#Features selected through correlation
y=data['']#Target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 0.5],
    'max_depth': [5, 10, 15, None],
    'max_leaf_nodes': [None, 10, 20, 30],
}

rf_model=RandomForestRegressor(random_state=42,oob_score=True)

kf=KFold(n_splits=5)

grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=kf,
    n_jobs=-1,
    scoring='r2'
)

grid_search.fit(X_train,y_train)

best_params=grid_search.best_params_
best_n_estimators=best_params['n_estimators']
print(best_params)

best_rf_model=RandomForestRegressor(
    random_state=42,
    oob_score=True,
    warm_start=True,
    **best_params
)

train_r2=[]
test_r2=[]
train_rmse=[]
test_rmse=[]

for i in range(1,best_n_estimators+1):
    best_rf_model.set_params(n_estimators=i)
    best_rf_model.fit(X_train,y_train)

    train_pred=best_rf_model.predict(X_train)
    test_pred=best_rf_model.predict(X_test)
    train_r2_val=r2_score(y_train,train_pred)
    test_r2_val=r2_score(y_test,test_pred)
    train_r2.append(train_r2_val)
    test_r2.append(test_r2_val)

    train_rmse_val=np.sqrt(mean_squared_error(y_train,train_pred))
    test_rmse_val=np.sqrt(mean_squared_error(y_test,test_pred))
    train_rmse.append(train_rmse_val)
    test_rmse.append(test_rmse_val)

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 2)
plt.plot(range(1,best_n_estimators+1),train_r2,label='Train R²',color='blue')
plt.plot(range(1,best_n_estimators+1),test_r2,label='Test R²',color='green')
plt.xlabel('Number of Trees')
plt.ylabel('R²')
plt.title('R² vs Number of Trees')
plt.legend()
plt.grid(True)
plt.subplot(2,2,3)
plt.plot(range(1,best_n_estimators+1),train_rmse,label='Train RMSE',color='blue')
plt.plot(range(1,best_n_estimators+1),test_rmse,label='Test RMSE',color='green')
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of Trees')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=211)

# Use the defined function to train the model and get the scores of both the training and the test set
model_rf=RandomForestRegressor(n_estimators=70,min_samples_split=3,min_samples_leaf=1,max_features='sqrt', max_depth=10, max_leaf_nodes=100, random_state=42)
model_rf.fit(X_train,y_train)
joblib.dump(model_rf,"..\\rf_model_.pkl")

y_train_predict=model_rf.predict(X_train)
y_test_predict=model_rf.predict(X_test)
train_rmse=np.around(np.sqrt(mean_squared_error(y_train,y_train_predict)),decimals=3)
test_rmse=np.around(np.sqrt(mean_squared_error(y_test,y_test_predict)),decimals=3)
train_r2=np.around(r2_score(y_train,y_train_predict),decimals=3)
test_r2=np.around(r2_score(y_test,y_test_predict),decimals=3)
print("The RMSE on training set is {}".format(train_rmse))
print("The RMSE on test set is {}".format(test_rmse))
print("R2 score on training set is {}".format(train_r2))
print("R2 score on test set is {}".format(test_r2))