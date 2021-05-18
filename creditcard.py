import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_repor

W_Data=pd.read_csv("../input/creditcard.csv")
W_Data.dropna(thresh=284315)
Data=W_Data

#Exploring the Dataset
Data.sample(frac=0.1).head(n=5)
#From the above it could be infered that the dataset 
#has 28 anonymized features and 2 non anonymized features i)
# Amount and ii) Class (whether the transcation was a fraud or not)


#8 rows × 31 columns
Data.describe()


#Spliting data
Positives=W_Data[W_Data['Class']==1]
Negatives=W_Data[W_Data['Class']==0]

print((len(Positives)/len(W_Data))*100,"%")#0.1727485630620034 %


#plot
sns.kdeplot(Positives['Amount'],shade=True,color="red")

sns.kdeplot(Negatives['Amount'],shade=True,color="green")

#Non-Fradulent Data
sns.kdeplot(Negatives['Time'],shade=True,color="red")

#Fradulent Data
sns.kdeplot(Positives['Time'],shade=True,color="green")

'''

For the purpose of evaluating algorithms. Lets first evaluate 
them on a part of the data since running the algorithms on 
all 284315 samples would be cumbersome. Lets take 50,000 examples
 for the purpose of evaluating the algorithms. 
 We will use the entire dataset to evaluate the final accuracy.
 '''
 
Negatives=Data[Data['Class']==0]
Positives=Data[Data['Class']==1]

Train_Data=Data[1:50000]
Target=Train_Data['Class']
Train_Data.drop('Class',axis=1,inplace=True)
Negatives=Data[Data['Class']==0]

x_train,x_test,y_train,y_test=train_test_split(Train_Data,Target,test_size=0.5,random_state=0)

#Support Vector Machine
clf_l=svm.SVC(kernel='linear')
clf_l.fit(x_train,y_train)
print(classification_report(y_test,clf_l.predict(x_test)))
Positives=Data[Data['Class']==1]

'''             precision    recall  f1-score   support

          0       1.00      1.00      1.00     24931
          1       0.64      0.67      0.65        69

avg / total       1.00      1.00      1.00     2500
'''

#Random Forest Classifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
print(classification_report(y_test,clf.predict(x_test))
      
'''
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     24931
          1       0.73      0.78      0.76        69

avg / total       1.00      1.00      1.00     25000
'''

E_Data=pd.read_csv("../input/creditcard.csv")
E_Data.dropna(thresh=284315)
E_Train_Data=E_Data
E_Target=E_Train_Data['Class']
E_Train_Data.drop('Class',axis=1,inplace=True)
x_train_E,x_test_E,y_train_E,y_test_E=train_test_split(E_Train_Data,E_Target,test_size=0.5,random_state=0)

clf_E = RandomForestClassifier(max_depth=2, random_state=0)
clf_E.fit(x_train, y_train)
print(classification_report(y_test_E,clf_E.predict(x_test_E)))

'''

             precision    recall  f1-score   support

          0       1.00      1.00      1.00    142161
          1       0.87      0.47      0.61       243

avg / total       1.00      1.00      1.00    142404
'''

#Anomaly Detection Algorithms

#Anomaly Detection

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

#One Class SVM
W_Data=pd.read_csv("../input/creditcard.csv")
W_Data.dropna(thresh=284315)
Data=W_Data[1:50000]

W_Data=pd.read_csv("../input/creditcard.csv")
W_Data.dropna(thresh=284315)
Data=W_Data[1:50000]

Negatives=Data[Data['Class']==0]
Positives=Data[Data['Class']==1]

#RBF Kernel
clf_AD = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf_AD.fit(Negatives)

'''
OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.1, kernel='rbf',
max_iter=-1, nu=0.1, random_state=None, shrinking=True, tol=0.001,
verbose=False)
'''
#Linear Kernel
clf_AD_L = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=0.1)
clf_AD_L.fit(Negatives)

'''
OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.1, kernel='linear',
      max_iter=-1, nu=0.1, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
'''

IFA=IsolationForest()
IFA.fit(Negatives)

'''
IsolationForest(bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None,
        verbose=0)
'''
train_AD_L=clf_AD_L.predict(Negatives)
test_AD_L=clf_AD_L.predict(Positives)

train_IFA=IFA.predict(Negatives)
test_IFA=IFA.predict(Positives)

train_AD=clf_AD.predict(Negatives)
test_AD=clf_AD.predict(Positives)

def Train_Accuracy(Mat):
   
   Sum=0
   for i in Mat:
    
        if(i==1):
        
           Sum+=1.0
            
   return(Sum/len(Mat)*100)

def Test_Accuracy(Mat):
   
   Sum=0
   for i in Mat:
    
        if(i==-1):
        
           Sum+=1.0
            
   return(Sum/len(Mat)*100)

print("Training: One Class SVM (RBF) : ",(Train_Accuracy(train_AD)),"%")
print("Test: One Class SVM (RBF) : ",(Test_Accuracy(test_AD)),"%")

'''

Training: One Class SVM (RBF) :  96.90606005897575 %
Test: One Class SVM (RBF) :  92.0 %
'''

print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")

'''
Training: Isolation Forest:  95.9981946199675 %
Test: Isolation Forest:  89.94594594594594 %
'''
print("Training: One Class SVM (Linear) : ",(Train_Accuracy(train_AD_L)),"%")
print("Test: One Class SVM (Linear) : ",(Test_Accuracy(test_AD_L)),"%")

'''
Training: One Class SVM (Linear) :  89.9981946199675 %
Test: One Class SVM (Linear) :  81.027027027027027 %
'''

W_Data=pd.read_csv("../input/creditcard.csv")
W_Data.dropna(thresh=284315)
Data=W_Data
Positives_E=W_Data[W_Data['Class']==1]
Negatives_E=W_Data[W_Data['Class']==0]

IFA=IsolationForest()
IFA.fit(Negatives_E)
train_IFA=IFA.predict(Negatives)
test_IFA=IFA.predict(Positives)

print("Training: Isolation Forest: ",(Train_Accuracy(train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(test_IFA)),"%")

'''
Training: Isolation Forest:  92.53984874927283 %
Test: Isolation Forest:  90.94594594594594 %
'''
#plot correlation matrix
plt.figure(figsize=(20,18))
Corr=Data[Data.columns].corr()
sns.heatmap(Corr,annot=True
           
#Evaluating Models on OverSampled Data

from imblearn.over_sampling import SMOTE 

W_Data=pd.read_csv("../input/creditcard.csv")
W_Data.dropna(thresh=284315)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(W_Data, W_Data['Class'])


S_Positives=[]
S_Negatives=[]
for i in range(0,len(X_res)):
    if(y_res[i]==0):
        S_Negatives.append(X_res[i])
    else:
        S_Positives.append(X_res[i])            
        
print("Training: Isolation Forest: ",(Train_Accuracy(S_train_IFA)),"%")
print("Test: Isolation Forest: ",(Test_Accuracy(S_test_IFA)),"%")
Training: Isolation Forest:  89.99982413871938 %
Test: Isolation Forest:  90.33360884933964 %

E_Data=pd.read_csv("../input/creditcard.csv")
E_Data.dropna(thresh=284315)
Outcome=E_Data['Class']
E_Data.drop('Class',axis=1,inplace=True)
X_res, y_res = sm.fit_sample(E_Data,Outcome)
x_train_E,x_test_E,y_train_E,y_test_E=train_test_split(X_res,y_res,test_size=0.5,random_state=0)
x_train_O,x_test_O,y_train_O,y_test_O=train_test_split(E_Data,Outcome,test_size=0.5,random_state=0)
:
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train_E, y_train_E)
print(classification_report(y_test_O,clf.predict(x_test_O)))

'''
             precision    recall  f1-score   support

          0       1.00      1.00      1.00    142161
          1       0.23      0.83      0.36       243

avg / total       1.00      0.99      1.00    142404

'''
