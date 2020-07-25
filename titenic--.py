# -*- coding: utf-8 -*-
"""
Titenic project """

#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
# Categorical encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import category_encoders as ce

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, accuracy_score, balanced_accuracy_score, explained_variance_score


# importing dataset

dataset1=pd.read_csv('train.csv')
dataset2=pd.read_csv('test.csv')



#x1=dataset.iloc[].values

train_NA = dataset1.isna().sum()
test_NA = dataset2.isna().sum()

pd.concat([train_NA, test_NA], axis=1, sort = False, keys=['Train NA', 'Test NA']).transpose()
# splitted training and testing data
y_train=dataset1.Survived
X_train=dataset1[dataset1.columns.drop('Survived')]
X_test=dataset2



#combine parent/children && sibling/spouse into family

def alone(df):
    if ('SibSp' in df.columns) and ('Parch' in df.columns):
        df['Family'] = df['SibSp'] + df['Parch'] + 1
    
    le = LabelEncoder()
    df['Ticket'] = le.fit_transform(df['Ticket'])
    df['Same_Ticket'] = df.duplicated(['Ticket'])
    
    df['Alone'] = np.where((df['Family'] > 1) | (df['Same_Ticket']), False, True)
    plt.figure(figsize=(20, 12))
    sns.catplot(x="Alone", kind="count", palette="ch:.25", data=df)
    plt.title('Number of passengers travelling alone or not')
    
    df = df[df.columns.drop('SibSp', errors='ignore')]
    df = df[df.columns.drop('Parch', errors='ignore')]
    df = df[df.columns.drop('Ticket', errors='ignore')]
    df = df[df.columns.drop('Same_Ticket', errors='ignore')]
    return df

X_train = alone(X_train)
X_test= alone(X_test)
# number of people per cabin

def fix_cabin(df):
    t = df.Cabin.fillna('U')
    df['Cabin'] = t.str.slice(0,1)
    
    plt.figure(figsize=(20, 15))
    sns.catplot(x="Cabin", kind="count", palette="ch:.25", data=df)
    plt.title('Number of passengers per cabin')
    
    le = LabelEncoder()
    df['Cabin'] = le.fit_transform(df['Cabin'])
fix_cabin(X_train)
fix_cabin(X_test)

#FILLING missing embarked value with the most frequent value

def fix_embark(df):
    most_freq = df.Embarked.mode().iloc[0]
    df['Embarked'] = df.Embarked.fillna(most_freq)
    plt.figure(figsize=(20, 12))
    sns.catplot(x="Embarked", kind="count", palette="ch:.25", data=df)
    plt.title('Number of passengers per embark location')
fix_embark(X_train)
fix_embark(X_test)
# changing categorical features using onehotencoding

def encode_nominal(df, cols):
    oh_encoder = OneHotEncoder()
    oh_cols = pd.DataFrame(oh_encoder.fit_transform(df[cols]).toarray())
    oh_cols.columns = oh_encoder.get_feature_names(cols)
    oh_cols.index = df.index
    df = df.join(oh_cols)
    df = df.drop(columns=cols)
    return df
X_train= encode_nominal(X_train, ['Embarked', 'Sex', 'Alone'])
X_test= encode_nominal(X_test, ['Embarked', 'Sex', 'Alone'])
X_train.head()
# name to title
def extract_title(df):
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0].str.strip()
        df = df[df.columns.drop('Name')]
        
    sns.catplot(kind="count", y="Title", palette="ch:.25", data=df, order=df['Title'].value_counts().index)
    plt.title('Number of passengers per Title')
    le = LabelEncoder()
    df['Title'] = le.fit_transform(df['Title'])
    return df, le
X_train, le = extract_title(X_train)
X_test, le = extract_title(X_test)
# categorical to numerical using labelencoder
def convert_cat(df):
    le = LabelEncoder()

    le_train_X = df.copy()

    # Encode categorical features
    s = df.dtypes=='object'
    cat_features = list(s[s].index)

    for col in cat_features:
        le_train_X[col] = le.fit_transform(df[col])

    return le_train_X
# predicting missing age data
    
from sklearn.impute import SimpleImputer
X_train1=X_train.iloc[:,:].values
X_test1=X_test.iloc[:,:].values
imp=SimpleImputer(missing_values=np.nan, strategy='mean',verbose=0)
imp1=SimpleImputer(missing_values=np.nan, strategy='mean',verbose=0)
imp=imp.fit(X_train1[:,:])
imp1=imp1.fit(X_test1[:,:])
X_train1[:,:]=imp.transform(X_train1[:,:])
X_test1[:,:]=imp1.transform(X_test1[:,:])
X_train=pd.DataFrame(X_train1)
X_test=pd.DataFrame(X_test1)
X_req=X_test1[:,(0)]
X_req=pd.DataFrame(X_req)
X_req['PassengerId']=X_req
"""X_test['DataFrame Column'] = X_test['DataFrame Column'].apply(str)
X_test['DataFrame Column'] = X_test['DataFrame Column'].astype(str)
X_test =X_test.applymap(str)"""

#X_new=X_test[:,0:1]

#predict by logistic regression model"""

"""from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)"""
#debugg
x=5
print(x)

#feature scaling
from sklearn import preprocessing
#min_max_scaler
X_train=X_train.iloc[:,:].values
X_test=X_test.iloc[:,:].values
min_max=preprocessing.MinMaxScaler(feature_range=(0,1))
X_train=min_max.fit_transform(X_train)
X_test=min_max.transform(X_test)

# standardisation
standard=preprocessing.StandardScaler()
X_train=standard.fit_transform(X_train)
X_test=standard.transform(X_test)


#predict by randomforest classifier
#X_train=X_train.iloc[:,:].values
#y_train=y_train.iloc[:,:].values
#X_test=X_test.iloc[:,:].values
classifier1=RandomForestClassifier(n_estimators=50,random_state=0)
classifier1.fit(X_train,y_train)
y_pred=classifier1.predict(X_test)
#X_test=X_test.iloc[:,:].values
#X_test=X_test[0:10,0:14]
#X_test2=X_test.copy()
# Shape of training data (num_rows, num_columns)
x=5
print(x)
#print(X_train.shape)"""

# Number of missing values in each column of training data


#y_pred=classifier1.predict(X_test)
# output

output = pd.DataFrame({
    "PassengerId": X_req["PassengerId"],
    "Survived" : y_pred
})
output.to_csv('gender_submission.csv', index=False)