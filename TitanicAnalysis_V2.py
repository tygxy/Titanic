# coding:utf-8

import os
import pandas as pd 
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import metrics
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

sns.set(style='whitegrid',color_codes=True)

# 0.导入数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1.数据展示

# 1.1 Survived
# print train['Survived'].value_counts(normalize=True)
# sns.countplot(train['Survived'])
# plt.show()

# 1.2 Pclass
# print train['Survived'].groupby(train['Pclass']).mean()
# sns.countplot(train['Pclass'],hue=train['Survived'])
# plt.show()

# 1.3 Name
# train['Name_Title'] = train['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
# print train['Name_Title'].value_counts()
# print train['Survived'].groupby(train['Name_Title']).mean()
# train['Name_Len'] = train['Name'].apply(lambda x:len(x))
# print train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()
# print pd.qcut(train['Name_Len'],5).value_counts()

# 1.4 Sex
# print train['Survived'].groupby(train['Sex']).mean()

# 1.5 Age
# print train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()
# sns.countplot(pd.qcut(train['Age'],5),hue=train['Survived'])
# plt.show()

# 1.6 Ticket
# train['Ticket_Len'] = train['Ticket'].apply(lambda x:len(x))
# train['Ticket_Lett'] = train['Ticket'].apply(lambda x:str(x)[0])
# print train['Survived'].groupby(train['Ticket_Lett']).mean()



# 2. 数据预处理
def names(train,test):
	for i in [train,test]:
		i['Name_Len'] = i['Name'].apply(lambda x:len(x))
		i['Name_Title'] = i['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x:x.split()[0])
		del i['Name']
	return train,test

def age_impute(train,test):
	for i in [train,test]:
		i['Age_Null_Flag'] = i['Age'].apply(lambda x:1 if pd.isnull(x) else 0)
		data = train.groupby(['Name_Title','Pclass'])['Age']
		i['Age'] = data.transform(lambda x:x.fillna(x.mean()))
	return train,test

def fam_size(train,test):
	for i in [train,test]:
		i['Fam_Size'] = np.where((i['SibSp']+i['Parch'])==0,'Solo',
							np.where((i['SibSp']+i['Parch']) <=3,'Nuclear','Big'))
		del i['SibSp']
		del i['Parch']
	return train,test

def ticket_grouped(train,test):
	for i in [train,test]:
		i['Ticket_Lett'] = i['Ticket'].apply(lambda x:str(x)[0])
		i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x:str(x))
		i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1','2','3','S','P','C','A']),i['Ticket_Lett'],
			                 np.where((i['Ticket_Lett']).isin(['W','4','7','6','L','5','8']),
			                 	'Low_ticket','Other_ticket'))
		i['Ticket_Len'] = i['Ticket'].apply(lambda x:len(x))
		del i['Ticket']
	return train,test

def cabin(train,test):
	for i in [train,test]:
		i['Cabin_Letter'] = i['Cabin'].apply(lambda x:str(x)[0])
		del i['Cabin']
	return train,test

def cabin_num(train,test):
	for i in [train,test]:
		i['Cabin_num1'] = i['Cabin'].apply(lambda x:str(x).split(' ')[-1][1:])
		i['Cabin_num1'].replace('an',np.NaN,inplace=True)
		i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x:int(x) if not pd.isnull(x) and x != '' else np.NaN)
		i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)

	train = pd.concat((train,pd.get_dummies(train['Cabin_num'],prefix='Cabin_num')),axis=1)
	test = pd.concat((test,pd.get_dummies(test['Cabin_num'],prefix='Cabin_num')),axis=1)
	del train['Cabin_num1']
	del test['Cabin_num1']
	del train['Cabin_num']
	del test['Cabin_num']
	return train,test

def embarked_impute(train,test):
	for i in [train,test]:
		i['Embarked'] = i['Embarked'].fillna('S')
	return train,test

def dummies(train,test,columns=['Pclass','Sex','Embarked','Ticket_Lett','Cabin_Letter','Name_Title','Fam_Size']):
	for column in columns:
		train[column] = train[column].apply(lambda x:str(x))
		test[column] = test[column].apply(lambda x:str(x))
		good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
		train = pd.concat((train,pd.get_dummies(train[column],prefix=column)[good_cols]),axis=1)
		test = pd.concat((test,pd.get_dummies(test[column],prefix=column)[good_cols]),axis=1)
		del train[column]
		del test[column]
	return train,test

def drop(train):
	del train['PassengerId']
	return train

def scaler(train,test):
	scaler = preprocessing.StandardScaler()
	age_scale_param = scaler.fit(train['Age'])
	train['Age'] = scaler.fit_transform(train['Age'],age_scale_param)
	test['Age'] = scaler.fit_transform(test['Age'],age_scale_param)
	fare_scale_param = scaler.fit(train['Fare'])
	train['Fare'] = scaler.fit_transform(train['Fare'],fare_scale_param)
	test['Fare'] = scaler.fit_transform(test['Fare'],age_scale_param)
	return train,test


train,test = names(train,test)
train,test = age_impute(train,test)
train,test = fam_size(train,test)
train,test = ticket_grouped(train,test)
train,test = cabin_num(train,test)
train,test = cabin(train,test)
train,test = embarked_impute(train,test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train,test = scaler(train,test)
train,test = dummies(train,test)
train = drop(train)

# 3 超参数调整

# 4 模型预测

# 4.1 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini',n_estimators=700,min_samples_split=10,
							min_samples_leaf=1,max_features='auto',oob_score=True,random_state=1,n_jobs=-1)
rf.fit(train.iloc[:,1:],train.iloc[:,0])

predictions= rf.predict(test.iloc[:,1:])
result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('predictions_rf.csv',index=False)

# 参数重要性排行 
# print pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
#            pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
#           axis = 1).sort_values(by='importance', ascending = False)[:20]




