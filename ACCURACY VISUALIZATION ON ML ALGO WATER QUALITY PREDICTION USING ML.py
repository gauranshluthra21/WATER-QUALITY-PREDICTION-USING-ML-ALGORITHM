#!/usr/bin/env python
# coding: utf-8

# MACHINE LEARNING ALGORITHMS : 

# 1. LOGISTIC REGRESSION 
# 2. DECISION TREE
# 3. RANDOM FOREST
# 4. K-NEAREST NEIGHBOURS 
# 5. SUPPORT VECTOR MACHINE
# 6. ADABooST
# 7. Bagging
# 8. Perceptron

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\DATA ANALYSIS PROJECT\water_potability.csv")


# In[4]:


df


# In[ ]:





# In[5]:


df.head()


# In[6]:


df.columns


# In[7]:


df.describe


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


plt.figure(figsize=(12,8))
sns.heatmap(df.isnull())


# In[11]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True)


# In[12]:


sns.countplot(x="Potability",data=df)


# In[13]:


df["Potability"].value_counts()


# In[14]:


#visualization dataset also checking for outliers

fig, ax=plt.subplots(ncols=5,nrows=2,figsize=(20,10))
ax=ax.flatten()

index=0

for col,values in df.items():
    sns.boxplot(y=col,data=df,ax=ax[index])
    index +=1


# In[15]:


sns.pairplot(df)


# In[116]:


df.isnull().mean().plot.bar(figsize=(12,8))
plt.xlabel("FEATURES")
plt.ylabel("PERCENTAGE OF MISSING VALUES")


# In[117]:


df["ph"] = df["ph"].fillna(df["ph"].mean())
df["Sulfate"]=df["Sulfate"].fillna(df["Sulfate"].mean())
df["Trihalomethanes"]=df["Trihalomethanes"].fillna(df["Trihalomethanes"].mean())


# In[18]:


df.isnull().sum()


# In[19]:


sns.heatmap(df.isnull())


# In[20]:


df.head()


# In[21]:


x=df.drop("Potability",axis=1)
y=df["Potability"]


# In[22]:


x.shape,y.shape


# In[23]:


scaler=StandardScaler()
x=scaler.fit_transform(x)
x


# In[118]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[119]:


x_train.shape,x_test.shape


# In[ ]:





# # LOGISTIC REGRESSION

# In[120]:


from sklearn.linear_model import LogisticRegression

#object
model_lr=LogisticRegression()


# In[121]:


#Training of the model

model_lr.fit(x_train,y_train)


# In[122]:


#making prediction
pred_lr=model_lr.predict(x_test)


# In[123]:


#accuracy score

accuracy_score_lr=accuracy_score(y_test,pred_lr)
accuracy_score_lr*100


# In[124]:


cm1=confusion_matrix(y_test,pred_lr)
cm1


# In[125]:


f1score = f1_score(y_test, pred_lr, average='weighted')

print("F1-score:", f1score)


# In[126]:


recall = recall_score(y_test,pred_lr)

print("Recall:", recall)


# In[ ]:





# In[127]:


mse = mean_squared_error(y_test, pred_lr)


# In[128]:


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[129]:


#ROC area

probs = model_lr.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[130]:


# Predict on test data
pred_lr=model_lr.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_lr)

print("Precision score:", precision)


# In[ ]:





# # DECISION TREE CLASSIFIER

# In[131]:


from sklearn.tree import DecisionTreeClassifier

#creating the model object
model_dt=DecisionTreeClassifier(max_depth=4)


# In[132]:


#Training of decision tree

model_dt.fit(x_train,y_train)


# In[133]:


#Making prediction using Decision Tree

pred_dt=model_dt.predict(x_test)


# In[134]:


accuracy_score_dt=accuracy_score(y_test,pred_dt)
accuracy_score_dt*100


# In[135]:


#confusion matrix

cm2=confusion_matrix(y_test,pred_dt)
cm2


# In[136]:


f2score = f1_score(y_test, pred_dt, average='weighted')

print("F1-score:", f1score)


# In[137]:


recall2 = recall_score(y_test,pred_dt)

print("Recall:", recall)


# In[138]:


mse = mean_squared_error(y_test,pred_dt)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[139]:


#ROC area

probs = model_dt.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[140]:


# Predict on test data
pred_dt=model_dt.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_dt)

print("Precision score:", precision)


# # RANDOM FOREST CLASSIFIER

# In[141]:


from sklearn.ensemble import RandomForestClassifier

#creating the model object
model_rf=RandomForestClassifier(max_depth=4)


# In[142]:


#training the model

model_rf.fit(x_train,y_train)


# In[143]:


#making predictions

pred_rf = model_rf.predict(x_test)


# In[144]:


accuracy_score_rf=accuracy_score(y_test,pred_dt)
accuracy_score_rf*100


# In[145]:


cm3=confusion_matrix(y_test,pred_rf)
cm3


# In[146]:


f3score = f1_score(y_test, pred_rf, average='weighted')

print("F1-score:", f1score)


# In[147]:


recall3 = recall_score(y_test,pred_rf)

print("Recall:", recall)


# In[148]:


mse = mean_squared_error(y_test,pred_rf)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[149]:


#ROC area

probs = model_rf.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[150]:


# Predict on test data
pred_lr=model_rf.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_rf)

print("Precision score:", precision)


# # K-NEIGHBORS (KNN)

# In[151]:


from sklearn.neighbors import KNeighborsClassifier

#creating model object
model_knn=KNeighborsClassifier()


# In[152]:


for i in range(4,12):
    model_knn=KNeighborsClassifier(n_neighbors=i)
    model_knn.fit(x_train,y_train)
    pred_knn=model_knn.predict(x_test)
    accuracy_score_knn=accuracy_score(y_test,pred_knn)
    print(i,accuracy_score_knn)


# In[153]:


model_knn=KNeighborsClassifier(n_neighbors=11)
model_knn.fit(x_train,y_train)
pred_knn=model_knn.predict(x_test)
accuracy_score_knn=accuracy_score(y_test,pred_knn)
print(accuracy_score_knn*100)


# In[154]:


cm4=confusion_matrix(y_test,pred_knn)
cm4


# In[155]:


f4score = f1_score(y_test, pred_knn, average='weighted')

print("F1-score:", f1score)


# In[156]:


recall4 = recall_score(y_test,pred_knn)

print("Recall:", recall)


# In[157]:


mse = mean_squared_error(y_test,pred_knn)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[158]:


#ROC area

probs = model_knn.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[159]:


# Predict on test data
pred_lr=model_knn.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_knn)

print("Precision score:", precision)


# # SUPPORT VECTOR MACHINE

# In[160]:


from sklearn.svm import SVC

#creating object of model
model_svm=SVC(kernel="rbf",probability=True)


# In[161]:


#model training

model_svm.fit(x_train,y_train)


# In[162]:


#make prediction

pred_svm=model_svm.predict(x_test)


# In[163]:


accuracy_score_svm=accuracy_score(y_test,pred_svm)
accuracy_score_svm*100


# In[164]:


cm5=confusion_matrix(y_test,pred_svm)
cm5


# In[165]:


f5score = f1_score(y_test, pred_svm, average='weighted')

print("F1-score:", f1score)


# In[166]:


recall5 = recall_score(y_test,pred_svm)

print("Recall:", recall)


# In[167]:


mse = mean_squared_error(y_test,pred_svm)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[168]:


#ROC area

probs = model_svm.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[169]:


# Predict on test data
pred_lr=model_knn.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_knn)

print("Precision score:", precision)


# # ADABOOST CLASSIFIER

# In[170]:


from sklearn.ensemble import AdaBoostClassifier

#making object of model
model_ada=AdaBoostClassifier(n_estimators=200,learning_rate=0.03)


# In[171]:


#Training of the model
model_ada.fit(x_train,y_train)


# In[172]:


#making prediction
pred_ada=model_ada.predict(x_test)


# In[173]:


#accuracy check 
accuracy_score_ada=accuracy_score(y_test,pred_ada)
accuracy_score_ada*100


# In[174]:


cm6=confusion_matrix(y_test,pred_ada)
cm6


# In[175]:


f6score = f1_score(y_test, pred_ada, average='weighted')

print("F1-score:", f1score)


# In[176]:


recall6 = recall_score(y_test,pred_ada)

print("Recall:", recall)


# In[177]:


mse = mean_squared_error(y_test,pred_ada)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[178]:


#ROC area

probs = model_ada.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[179]:


# Predict on test data
pred_lr=model_ada.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, pred_ada)

print("Precision score:", precision)


# # BAGGING ALGORITHM

# In[180]:


from sklearn.ensemble import BaggingClassifier


# In[181]:


bag_model = BaggingClassifier(
base_estimator=DecisionTreeClassifier(), 
n_estimators=100, 
max_samples=0.8, 
bootstrap=True,
oob_score=True,
random_state=0
)


# In[182]:


bag_model.fit(x_train, y_train)


# In[183]:


y_pred_bag = bag_model.predict(x_test)


# In[184]:


bag_model.oob_score_*100


# In[185]:


cm7=confusion_matrix(y_test,y_pred_bag)
cm7


# In[186]:


f7score = f1_score(y_test, y_pred_bag, average='weighted')

print("F1-score:", f1score)


# In[187]:


recall7 = recall_score(y_test,y_pred_bag)

print("Recall:", recall)


# In[188]:


mse = mean_squared_error(y_test,y_pred_bag)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[189]:


#ROC area

probs = bag_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[190]:


# Predict on test data
y_pred_bag = bag_model.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, y_pred_bag)

print("Precision score:", precision)


# # Perceptron

# In[191]:


from numpy import mean
from numpy import std
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[192]:


percep_model = ExtraTreesClassifier()


# In[193]:


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(percep_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')


# In[194]:


percep_model.fit(x_train, y_train)


# In[195]:


y_pred_percept = percep_model.predict(x_test)


# In[196]:


mean(n_scores)*100


# In[197]:


cm8=confusion_matrix(y_test,y_pred_percept)
cm8


# In[198]:


f8score = f1_score(y_test, y_pred_percept, average='weighted')

print("F1-score:", f1score)


# In[199]:


recall8 = recall_score(y_test,y_pred_percept)

print("Recall:", recall)


# In[200]:


mse = mean_squared_error(y_test,y_pred_percept)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.bar(['MSE'], [mse])
ax.set_title('Mean Squared Error')
plt.show()


# In[201]:


#ROC area

probs = percep_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (AUC={:.2f})'.format(auc))
plt.show()


# In[202]:


# Predict on test data
y_pred_percept = percep_model.predict(x_test)

# Calculate precision score
precision = precision_score(y_test, y_pred_percept)

print("Precision score:", precision)


# In[203]:


from sklearn.naive_bayes import GaussianNB


# In[204]:


nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)
nb_model


# In[ ]:





# In[205]:


y_pred_nb = nb_model.predict(x_test)
print(classification_report(y_test, y_pred_nb))


# In[ ]:





# In[206]:


import numpy as np
import matplotlib.pyplot as plt

# F-scores and their labels
fscores = [f1score,f2score,f3score,f4score,f5score,f6score,f7score,f8score]
labels = ['F1', 'F2', 'F3', 'F4','F5','F6','F7','F8']

# Random precision and recall values for each F-score
# precision = np.random.rand(len(fscores))
# recall = np.random.rand(len(fscores))

# Set the width of the bars
bar_width = 0.35

# Plot the bars
fig, ax = plt.subplots()
bar1 = ax.bar(np.arange(len(fscores)), fscores, bar_width, label='F-SCORES')
# bar2 = ax.bar(np.arange(len(fscores))+bar_width, recall, bar_width, label='Recall')

# Add labels, title, and legend
ax.set_xlabel('F-score')
ax.set_ylabel('Score')
ax.set_title('F-scores FOR DIFFERENT ALGORITHMS')
ax.set_xticks(np.arange(len(fscores))+bar_width/2)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.show()


# In[ ]:





# # FINAL RESULT

# In[ ]:





# In[207]:


import pandas as pd

data1 = {
  "MODEL": ["Logistic Regression","Decision Tree","Random Forest","KNN ","SVM","AdaBoost","Bagging","Perceptron"],
  "ACCURATE SCORE": [accuracy_score_lr,accuracy_score_dt,accuracy_score_rf,accuracy_score_knn,accuracy_score_svm,accuracy_score_ada,bag_model.oob_score_,mean(n_scores)]
}

#load data into a DataFrame object:
df = pd.DataFrame(data1)

print(df) 


# In[208]:


sns.barplot(x="ACCURATE SCORE",y="MODEL",data=data1)
data1.sort_values(by="ACCURATE SCORE",ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a dictionary to store classification reports for different ML algorithms
class_reports = {
    'KNN': {'Class 1': [0.74,0.75,0.74,1353], 'Class 2': [0.75, 0.70, 0.73, 50]},
    'NAIVE BAYES': {'Class 1': [0.85, 0.85, 0.85, 110], 'Class 2': [0.70, 0.75, 0.73, 60]},
    'Algorithm 3': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]},
    'Algorithm 4': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]},
    'Algorithm 3': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]},
    'Algorithm 3': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]},
    'Algorithm 3': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]},
    'Algorithm 3': {'Class 1': [0.75, 0.80, 0.77, 90], 'Class 2': [0.80, 0.65, 0.72, 70]}
}

# Convert classification reports to a DataFrame
df = pd.DataFrame.from_dict({(i,j): class_reports[i][j] for i in class_reports.keys() for j in class_reports[i].keys()},
                            orient='index', columns=['precision', 'recall', 'f1-score', 'support'])

# Transpose DataFrame to have one row for each class
df = df.unstack()

# Plot histogram for each class
for c in df.columns:
    df[c].plot.hist(alpha=0.5, bins=10, title=c)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




