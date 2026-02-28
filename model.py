from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


df = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/Training.csv')
sym_des = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/symtoms_df.csv')
precautions = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/precautions_df.csv')
workout = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/workout_df.csv')
description = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/description.csv')
medications = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/medications.csv')
diets = pd.read_csv('C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation/diets.csv')


#Split-the-Data-in-Train-Test
X = df.drop('prognosis' ,axis=1)
y = df['prognosis']

le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# split the data 
X_test,X_train,y_test,y_train = train_test_split(X,Y,test_size =0.2 , random_state=42)

#training model

models = {
      'SVC' : SVC(kernel='linear'),
      'Random Forest' :RandomForestClassifier(random_state=42,n_estimators=100),
      'KNeighbors' : KNeighborsClassifier(n_neighbors=5),
      'Gradient Boosting': GradientBoostingClassifier(random_state=42,n_estimators=100),
      'MultinomialNB' :MultinomialNB() 
}
for model_name , model in models.items():
    # Train model
    model.fit(X_train,y_train)
    # test model
    predictions = model.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test,predictions)
    # calculate confusion matrix
    cm = confusion_matrix(y_test,predictions)
    # print results
    print(f"{model_name} accuracy : {accuracy}")
    print(f"{model_name} confusion matrix :")
    print(np.array2string(cm,separator=', '))

#gives 100% accuracy so we go with SVC

svc = SVC(kernel='linear')

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_pred,y_test)

path = 'C:/Users/diyad/OneDrive/Desktop/Medicine_Recommendation//svc.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)

# Save the model
with open(path, 'wb') as file:
    pickle.dump(svc, file)

#load the data set
svc = pickle.load(open(path,'rb'))

# 2d arry convert
X_test.iloc[0].values.reshape(1,-1)

# test 1 :
print('Model Predictions :',svc.predict(X_test.iloc[0].values.reshape(1,-1)))
print('Actual Labels :', y_test[0])

# test 2 :
print('Model Predictions :',svc.predict(X_test.iloc[40].values.reshape(1,-1)))
print('Actual Labels :', y_test[40])


