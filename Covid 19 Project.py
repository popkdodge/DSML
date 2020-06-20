# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np 
import pandas as pd 
 


# %%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib 


# %%
dataset_url = 'https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19/master/dataset/Patient/PatientInfo.csv'


# %%
covid = pd.read_csv(dataset_url)


# %%
covid.head()


# %%
covid.tail()


# %%
covid.shape


# %%
covid.state.value_counts()


# %%
covid.isnull().sum()


# %%
covid['age'].value_counts()


# %%
covid['age'].value_counts().plot(kind='barh');


# %%
covid['age'].isnull().sum()


# %%
'''Replacing is null value for age with the median age of SK with Sk median age there is 40.8 and the age is catergorized in 10s so adding 14 40s into the null value of age''' 



# %%
covid.isnull().sum()


# %%
covid['state'].value_counts()


# %%
condition = {'isolated':0,'released':1,'deceased':3}
covid.state = covid.state.map(condition)


# %%
covid['province'].value_counts()


# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(sns.load_dataset('titanic').drop(columns=['alive']), random_state=0)
target = 'decease'


# %%
'''isolate=0 release='1' decease 0'''


# %%
covid['decease'] = covid.state == 3


# %%
covid.drop('decease',axis=1,inplace=True)


# %%
decease_stat = {1:0,2:1}
covid.decease = covid.decease.map(decease_stat)


# %%
covid['sex'].fillna('male', inplace = True)


# %%
inputs = covid[['sex','age','province']]
target = 'decease' 


# %%
from sklearn.preprocessing import LabelEncoder


# %%
sex = LabelEncoder()
age = LabelEncoder()
province = LabelEncoder()


# %%
features['sex_n'] = sex.fit_transform(inputs['sex'])
features['age_n'] = age.fit_transform(inputs['age'])
features['province_n'] = province.fit_transform(inputs['sex'])


# %%
covid[['sex','age','province']]


# %%
covid[['decease']]


# %%
inputs = covid[['sex','age','province']]
target = covid[['decease']]


# %%
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
le_age = LabelEncoder()
le_province = LabelEncoder()


# %%
inputs['sex_n'] = le_sex.fit_transform(inputs['sex'])
inputs['age_n'] = le_sex.fit_transform(inputs['age'])
inputs['province_n'] = le_sex.fit_transform(inputs['province'])
inputs.head()


# %%
inputs_n = inputs.drop(['sex','age','province'],axis='columns')
inputs_n


# %%
from sklearn import tree
model = tree.DecisionTreeClassifier()


# %%
model.fit(inputs_n,target)


# %%
model.score(inputs_n,target)


# %%
covid_19 = covid[['sex','age','province','decease']]


# %%
from sklearn.model_selection import train_test_split
train_test_split(X,y,test_size=0.2,)


# %%
y = covid_19.decease
X= covid_19.drop('decease', axis=1)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 123, stratify=y)


# %%
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))


# %%
hyperparameters = {'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__max_depth': [None, 5, 3, 1]}


# %%
clf = GridSearchCV(pipeline, hyperparameters, cv=10)


# %%
clf.fit(X_train, y_train)


# %%
X['male'] = X.age == 'male'
X['female'] = X.age == 'female'


# %%
X.drop('province',axis=1,inplace=True)


# %%
y = covid_19.decease


# %%
covid_19


# %%
X.province.value_counts()


# %%
X['66s'] = X.age == '66s'
X['100s'] = X.age == '100s'
X['0s'] = X.age == '0s'
X['90s'] = X.age == '90s'
X['10s'] = X.age == '10s'
X['80s'] = X.age == '80s'
X['70s'] = X.age == '70s'
X['60s'] = X.age == '60s'
X['30s'] = X.age == '30s'
X['40s'] = X.age == '40s'
X['50s'] = X.age == '50s'
X['20s'] = X.age == '20s'


# %%
X


# %%
X['Gyeongsangbuk-do'] = X.province == 'Gyeongsangbuk-do'
X['Gyeonggi-do'] = X.province == 'Gyeonggi-do'
X['Seoul'] = X.province == 'Seoul'
X['Chungcheongnam-do'] = X.province == 'Chungcheongnam-do'
X['Busan'] = X.province == 'Busan'
X['Gyeongsangnam-do'] = X.province == 'Gyeongsangnam-do'
X['Daegu'] = X.province == 'Daegu'
X['Incheon'] = X.province == 'Incheon'
X['Sejong'] = X.province == 'Sejong'
X['Chungcheongbuk-do'] = X.province == 'Chungcheongbuk-do'
X['Gangwon-do'] = X.province == 'Gangwon-do'
X['Gangwon-do'] = X.province == 'Gangwon-do'
X['Daejeon'] = X.province == 'Daejeon'
X['Gwangju'] = X.province == 'Gwangju'
X['Jeollabuk-do'] = X.province == 'Jeollabuk-do'
X['Jeju-do'] = X.province == 'Jeju-do'
X['Jeollanam-do'] = X.province == 'Jeollanam-do'


# %%
X.province.value_counts()


# %%
pred = clf.predict(X_test)
print (r2_score(y_test, pred))
print (mean_squared_error(y_test, pred))


# %%
model.predict


# %%
get_ipython().system('pip install graphviz')
get_ipython().system('apt-get install graphviz')


# %%
import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(model, out_file=None, feature_names=X, filled=True, rotate=True)
graphviz.Source(dot_data)


# %%


