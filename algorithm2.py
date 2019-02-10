
import numpy as np
import pandas as pd
import sklearn

import warnings
warnings.filterwarnings("ignore")


print("Launching Algorithm #2: LifeStyle NCDs")

"""#2: Life Style Diseases That You are Susceptible To (Non-Communical Diseases)

###access Data
"""

df_r = pd.read_csv('data/diseases.csv')
df_r.tail()

"""###Pre processing Data"""

columns = ['gender', 'employment_status', 'education', 'marital_status', 'children', 'avg_commute', 'daily_internet_use', 'available_vehicles', 'military_service', 'disease']
df_r1 = df_r[columns]


## Gender: Male and Female
m = {'male' : 1, 'female' : 0}
df_r1['gender'] = df_r1['gender'].map(m)
df_r1.rename(columns={'gender': 'male'}, inplace=True)


## Military Service: Yes No
m2 = {'yes' : 1, 'no' : 0}
df_r1['military_service'] = df_r1['military_service'].map(m2)


## Married: married or single
m3 = {'married': 1, 'single': 0}
df_r1['marital_status'] = df_r1['marital_status'].map(m3)


## Remove the disease of HIV/AIDS, endometriosis,
df_r1 = df_r1[df_r1['disease'] != 'HIV/AIDS']
df_r1 = df_r1[df_r1['disease'] != 'endometriosis']
df_r1 = df_r1[df_r1['disease'] != "Alzheimer's disease"]

df_r1.tail()
print("#############################")
print("Unique Diseases")
print(df_r1['disease'].unique())

"""####Turn Avg_Commute from minutes to Hours by div 60"""

df_r1['avg_commute'] = df_r1['avg_commute']/60
df_r1.head()

"""###Handling Categorical Data"""

# df_r1.dtypes

df_r1['employment_status'].value_counts()

"""####One Hot Encoding of Employment Status"""

df_r1 = pd.get_dummies(df_r1, columns=["employment_status"])
# df_r1.head()

"""####One Hot Encoding of Education"""

# df_r1['education'].value_counts()

df_r1 = pd.get_dummies(df_r1, columns=["education"])
# df_r1.head()

"""####Unique Diseases"""

# df_r1['disease'].value_counts()

"""####categorical Data Done Preprocessing"""

# df_r1.dtypes

"""###Remove Unnecessary *Features*"""

df_r1.drop(columns=['education_phD/MD', 'education_highscool', 'employment_status_unemployed', 'education_masters', 'education_phd/md'], inplace=True, axis=1)
# df_r1.head()

## Rename really long feature names to shorter ones
df_r1.rename(columns={'employment_status_employed': 'employed', 'employment_status_retired': 'retired', 'employment_status_student': 'student'}, inplace=True)

# df_r1.head()

# df_r1.shape

columns = df_r1.columns
# columns

"""### Shuffle Data"""

from sklearn.utils import shuffle
df_r1 = shuffle(df_r1)

"""###X / Y Split"""

X_r = df_r1[['male','marital_status','children','avg_commute','daily_internet_use','available_vehicles','military_service','employed','retired','student','education_bachelors','education_highschool']]
y_r = df_r1[['disease']]
X_r.head()
X_r.shape

"""###Train Test Split"""

length = len(X_r)
print("length of dataset is ", length)

test_size = int(length * 20/100)    ### 20 % will be dataset
print("test size", test_size)
## Split X
X_train_r = X_r[:length-test_size]
X_test_r = X_r[-test_size:]


## Split y
y_train_r = y_r[:length-test_size]
y_test_r = y_r[-test_size:]

###Create Classifier Model

"""####Classifier 2"""
### Currently a good Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=5), #, class_weight='balanced'),
    n_estimators=275,
    learning_rate=0.0125,
    algorithm="SAMME.R")

bdt.fit(X_train_r, y_train_r)

test_score = bdt.score(X_test_r, y_test_r)
train_score = bdt.score(X_train_r, y_train_r)

print("####################")
print("Model Evaluation")
print("####################")
print("Train Score is ", train_score)
print("Test Score is ", test_score)

print("End of Loading Algorithm #2")
print("##############################")

