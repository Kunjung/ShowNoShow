
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

df_s = pd.read_csv('noshow.csv')

"""###Preprocessin Data"""

columns = ['Gender', 'Age', 'Scholarship', 'SMS_received', 'Diabetes', 'Alcoholism', 'Handcap', 'Hipertension',	'No-show']
df_s1 = df_s[columns]
m = {'M' : 1, 'F' : 0}
df_s1['Gender'] = df_s1['Gender'].map(m)
df_s1.rename(columns={'Gender': 'Male'}, inplace=True)
df_s1.head()

"""###no show and show yes and no to 1 and 0"""

m2 = {'Yes' : 1, 'No' : 0}
df_s1['No-show'] = df_s1['No-show'].map(m2)
df_s1.head()

"""###Remove Age Column. It ain't useful. Also, remove SMS_received. it's only making things confusing"""

df_s1.drop(columns=['SMS_received'], inplace=True, axis=1)
df_s1.head()

len(df_s)

"""###Preprocessing Data No Show Yes No to 1 or 0

###randomize rows
"""

from sklearn.utils import shuffle
df_s1 = shuffle(df_s1)

"""###X and Y split"""

X = df_s1[['Age', 'Male', 'Scholarship', 'Diabetes', 'Alcoholism', 'Handcap', 'Hipertension']]
y = df_s1[['No-show']]

"""###Train Test Split"""

length = len(X)

test_size = int(length * 20/ 100)    ### 20 % will be dataset

## Split X
X_train = X[:length-test_size]
X_test = X[-test_size:]


## Split y
y_train = y[:length-test_size]
y_test = y[-test_size:]

"""### Making Model to train Model in"""

print("Training Begins")
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# clf = AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced'),
#                          algorithm="SAMME.R",
#                          n_estimators=150,
#                          )

# clf = DecisionTreeClassifier(class_weight='balanced', random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)


print("Training Ends")
print("#####################################")
score = clf.score(X_train, y_train)
print("Training Accuracy is ", score)


score = clf.score(X_test, y_test)
print("Test Accuracy is ", score)
print("#####################################")
print("#####################################")


print("Test Sample for No Shows")
X_noshows = X_test.loc[clf.predict(X_test) == 1]
print(X_noshows.shape)
print(X_noshows.head())

print("Test Sample for Yes Shows")
X_yesshows = X_test.loc[clf.predict(X_test) == 0]
print(X_yesshows.shape)