
## Streamlit for the app
import streamlit as st 

import numpy as np 

## Matplot for the charts/plots
import matplotlib.pyplot as plt

## Dataset from the Sklearn
from sklearn import datasets
## Classification model from the Sklearn
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

## App header or title
app_title='<p style="font-family:sans-serif; color:Blue; font-size: 42px;">Streamlit App-Dataset Classifier</p>'
st.markdown(app_title, unsafe_allow_html=True)

st.write(":blue[Explore the SKLEARN Dataset and use available models for the classification]")

dataset_name = st.sidebar.selectbox(
    ":red[Select one Dataset]",('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"###  Dataset Name: :green[{dataset_name}]")

classifier_name = st.sidebar.selectbox(
    ':red[Select one classifier]',('KNN', 'SVM', 'Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write(f'##### Shape of {dataset_name} dataset =', X.shape)
st.write(f'##### Number of groups for {dataset_name} dataset =', len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('depth range', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('estimators range', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

## Select and use the appropriate classification model as per the selection
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1)
    return clf

clf = get_classifier(classifier_name, params)

#### Classification using model ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = round(accuracy_score(y_test, y_pred), 3)

st.write(f'##### Classifier = :red[{classifier_name}]')
st.write(f'##### Accuracy= :red[{acc}]')


# Project the data with respect to 2 primary principal categories
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')

plt.xlabel('Category 1')
plt.ylabel('Category 2')
plt.colorbar()

st.pyplot(fig)
