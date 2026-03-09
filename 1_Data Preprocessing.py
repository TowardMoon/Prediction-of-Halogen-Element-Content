import pandas as pd
from fancyimpute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data=pd.read_excel('')

#Multiple Imputation by Chained Equations (MICE) imputation
imputer=IterativeImputer(max_iter=500,random_state=42)
data_imputed=imputer.fit_transform(data)


data_imputed=pd.DataFrame(data_imputed,columns=data.columns)

data_imputed.to_excel("..\\mice_data.xlsx",index=False,engine="openpyxl")

#Analyze the relationships
correlation_matrix=data_imputed.corr(method='pearson')
print(correlation_matrix)

#Due to the excessive number of features, we imported them into a table in order to find suitable features
correlation_matrix.to_excel("..\\mice_data.xlsx",index=False,engine="openpyxl")

plt.figure(figsize=(25, 25))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f",linewidths=.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


#SMOTE
data=pd.read_excel('..\\mice_data.xlsx')
X=data[['','','']]#Features selected through correlation
y=data['']#Target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=None)

#Define regression SMOTE function
def regression_smote(X,y,n_samples=100,k_neighbors=5):
    """
    :param X:Feature data
    :param y:Target value
    :param n_samples:Number of synthetic samples to generate(But if the input data is too small, it will be impossible to generate the desired amount of data.)
    :param k_neighbors:Number of nearest neighbors
    :return:Enhanced features and target values
    """
    X=np.array(X)
    y=np.array(y)

    nbrs=NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    _,indices=nbrs.kneighbors(X)

    new_X=[]
    new_y=[]

    for _ in range(n_samples):
        idx=np.random.randint(0,len(X))
        neighbor_idx=np.random.choice(indices[idx][1:])
        diff=X[neighbor_idx]-X[idx]
        gap=np.random.rand()
        new_X.append(X[idx]+gap*diff)
        new_y.append(y[idx]+gap*(y[neighbor_idx]-y[idx]))
    return np.array(new_X),np.array(new_y)

X_train_resampled,y_train_resampled=regression_smote(X_train,y_train,n_samples=50,k_neighbors=3)

X_train_augmented=np.vstack([X_train,X_train_resampled])
y_train_augmented=np.hstack([y_train,y_train_resampled])

print("Number of original training set samples：",len(X_train))
print("Number of training set samples after augmentation：",len(X_train_augmented))


#Evaluate the quality of SMOTE data
model=LinearRegression()
model.fit(X_train_augmented,y_train_augmented)
y_pred=model.predict(X_test)
score=r2_score(y_test,y_pred)
print(f"R²：{score:.4f}")

augmented_df=pd.DataFrame(X_train_augmented,columns=X.columns)
augmented_df[''] =y_train_augmented
augmented_df.to_excel("..\\smote_data.xlsx", index=False,engine='openpyxl')