## xpert_data_test
# Author: Javier Cuadrado Corz
# Date: 3 october 2019

import urllib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from sklearn.naive_bayes import GaussianNB

import xlrd

def load_data_file(file):
    """
    Load data from a link pointing to a excel file
    file: string with the path to file
    """
#     socket = urllib.request.urlopen(link)
    xd = pd.ExcelFile(link)
    df = xd.parse(xd.sheet_names[0], header=0)
    return df

def load_data_link(link):
    """
    Load data from a link pointing to a excel file
    link: string
    """
#     socket = urllib.request.urlopen(link)
    xd = pd.ExcelFile(link)
    df = xd.parse(xd.sheet_names[0], header=0)
    return df


def transform_data(df):
    """
    Performs the necessary transformations to the data. Transformations performed:
    * Drop rows with NA's
    * Trasnform "Basic_Salary" and "Bonus" variables from float to integers
    """
    
    df.dropna(axis=0, how='any', subset=['Basic_Salary', 'Bonus', 'Group'], inplace=True)
    df = df.astype({"Basic_Salary": int, "Bonus": int})
    
    return df


def select_features_target(df, features, target):
    """
    selects features and target
    df: dataframe
    features: list of column names
    target: string name of column target
    """
    
    X = df[features]
    y = df[target]
    
    return [X,y]
    

def preprocess_features(X, categorical_variables):
    """
    One-hot encoding
    X: dataframe with features or predictors
    categorical_variables: list
    
    returns numpy array
    """
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    preprocessor = ColumnTransformer(
        transformers = [('cat', ohe, categorical_variables)],
        remainder = 'passthrough'
    )
    
    X_t = preprocessor.fit_transform(X)
    
    return X_t


def train_test_data_split(X, y, test_size, random_state):
    """
    Split data in train and test
    test_size: float; 0 > test_size < 1
    
    returns list:
        [X_train, X_test, y_train, y_test]
    """
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, 
                         test_size = test_size,
                         random_state = random_state,
                         stratify=y)
    
    return [X_train, X_test, y_train, y_test]

def fit_gaussiannb(X_train, y_train, X_test, y_test):
    """
    Fits a Gaussian Naive Bayes classifiers and returns the f1 score
    """
    
    result = dict()
    gnb = GaussianNB()
    result['model'] = gnb.fit(X_train, y_train)
    
    result['predictions'] = gnb.predict(X_test)
    result['f1_score'] = f1_score(y_test, result['gnb_preds'], average='micro')
    
    return result