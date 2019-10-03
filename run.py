#!/usr/bin/python3

from xperthr_data_test import *

link = 'http://resources.xperthr.co.uk/surveys/salary/Sample/Work_Test_-_synthetic_data_ds.xlsx'
df = load_data(link)

df = transform_data(df)

X, y = select_features_target(df, ['Basic_Salary', 'Bonus', 'Group'], "Level")

X_t = preprocess_features(X, ["Group"])

X_train, X_test, y_train, y_test = train_test_data_split(X_t, y, test_size=0.3, random_state=123)

result = fit_gaussiannb(X_train, y_train, X_test, y_test)

print(result)

exit(0)
