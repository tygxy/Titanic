# coding:utf-8

import pandas as pd 
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

sns.set(style='whitegrid',color_codes=True)

# 0.导入数据
titanic_data = pd.read_csv('train.csv')

# 1.数据初认识
# print titanic_data.info()
# print titanic_data.describe()
# print titanic_data.isnull().sum()


# 2.数据图形展示

## 2.1 乘客属性的分布
# fig = plt.figure(figsize=(20,10))
# fig.set(alpha=0.2)

# plt.subplot2grid((3,3),(0,0))
# titanic_data.Survived.value_counts().plot(kind='bar')
# plt.title(u"获救情况(1为获救)")
# plt.ylabel(u"人数")

# plt.subplot2grid((3,3),(0,1))
# titanic_data.Pclass.value_counts().plot(kind='bar')
# plt.title(u"乘客等级分布")
# plt.ylabel(u"人数")

# plt.subplot2grid((3,3),(0,2))
# plt.scatter(titanic_data.Survived,titanic_data.Age)
# plt.grid(b=True,which='major',axis='y')
# plt.title(u"按年龄看获救分布")
# plt.ylabel(u"年龄")

# plt.subplot2grid((3,3),(1,0),colspan=2)
# titanic_data.Age[titanic_data.Pclass == 1].plot(kind='kde')
# titanic_data.Age[titanic_data.Pclass == 2].plot(kind='kde')
# titanic_data.Age[titanic_data.Pclass == 3].plot(kind='kde')
# # plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱',u'2等仓',u'3等仓'),loc='best')

# plt.subplot2grid((3,3),(1,2))
# titanic_data.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")

# plt.subplot2grid((3,3),(2,0),colspan=2)
# titanic_data.Age[titanic_data.Survived == 1].plot(kind='kde')
# titanic_data.Age[titanic_data.Survived == 0].plot(kind='kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.title(u"获救与没获救乘客年龄分布")
# plt.legend((u'获救',u'没获救'),loc='best')

# plt.show()

## 2.2 属性与获救结果的关联统计

### 2.2.1 乘客等级的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)

# Survived_0 = titanic_data.Pclass[titanic_data.Survived == 0].value_counts()
# Survived_1 = titanic_data.Pclass[titanic_data.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# df.plot(kind='bar',stacked=True)
# plt.title(u'不同乘客等级的获救情况')
# plt.xlabel(u'乘客等级')
# plt.ylabel(u'人数')
# plt.show()

### 2.2.2 乘客性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)

# Survived_0 = titanic_data.Sex[titanic_data.Survived == 0].value_counts()
# Survived_1 = titanic_data.Sex[titanic_data.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# df.plot(kind='bar',stacked=True)
# plt.title(u'不同乘客性别的获救情况')
# plt.xlabel(u'乘客性别')
# plt.ylabel(u'人数')
# plt.show()

### 2.2.3 各种舱级别情况下各性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)
# plt.title(u'根据舱等级和性别的获救情况')

# ax1=fig.add_subplot(141)
# titanic_data.Survived[titanic_data.Sex == 'female'][titanic_data.Pclass != 3].value_counts().plot(kind='bar',color='#FA2479')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'女性/高级舱'],loc='best')

# ax1=fig.add_subplot(142)
# titanic_data.Survived[titanic_data.Sex == 'female'][titanic_data.Pclass == 3].value_counts().plot(kind='bar',color='pink')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'女性/低级舱'],loc='best')

# ax1=fig.add_subplot(143)
# titanic_data.Survived[titanic_data.Sex == 'male'][titanic_data.Pclass != 3].value_counts().plot(kind='bar',color='lightblue')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'男性/高级舱'],loc='best')

# ax1=fig.add_subplot(144)
# titanic_data.Survived[titanic_data.Sex == 'male'][titanic_data.Pclass == 3].value_counts().plot(kind='bar',color='steelblue')
# ax1.set_xticklabels([u'获救',u'未获救'],rotation=0)
# ax1.legend([u'男性/低级舱'],loc='best')

# plt.show()

### 2.2.4 各登船港口的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)

# Survived_0 = titanic_data.Embarked[titanic_data.Survived == 0].value_counts()
# Survived_1 = titanic_data.Embarked[titanic_data.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救':Survived_1,u'未获救':Survived_0})
# df.plot(kind='bar',stacked=True)
# plt.title(u'不同乘客性别的获救情况')
# plt.xlabel(u'登陆港口')
# plt.ylabel(u'人数')
# plt.show()



# 3. 数据预处理

## 3.1 清洗掉无用的属性
titanic_data = titanic_data.drop(['PassengerId','Name','Ticket'],axis=1)

## 3.2 使用RandomForestRegressor补全缺失的年龄属性,填充Cabin和Embarked属性

def set_missing_ages(df):
	age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()

	y = known_age[:,0]
	X = known_age[:,1:]

	rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
	rfr.fit(X,y)

	predictiedAges = rfr.predict(unknown_age[:,1::])

	df.loc[(df.Age.isnull()),'Age'] =predictiedAges

	return df,rfr

def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
	df.loc[(df.Cabin.isnull()),'Cabin'] = "No"
	return df

def set_Embarked_type(df):
	df['Embarked'] = df['Embarked'].fillna('C')
	return df

data_train,rfr = set_missing_ages(titanic_data)
data_train = set_Cabin_type(data_train)
data_train = set_Embarked_type(data_train)

## 3.3 对类目性特征因子化
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df.drop(['Pclass','Sex','Cabin','Embarked'],axis=1,inplace=True)

## 3.4 将Age和Fare属性归一化
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)


# 4 机器学习模型

## 4.1 逻辑回归

### 4.1.1 整体训练
# train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_np = train_df.as_matrix()

# y = train_np[:,0]
# X = train_np[:,1:]

# clf = linear_model.LogisticRegression(C=1.0,tol=1e-6)
# clf.fit(X,y)

# from sklearn import neighbors 
# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=23)
# clf.fit(X,y)

## 4.1.2 交叉训练
# split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# clf = linear_model.LogisticRegression(C=1.0,tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:,1:])
# y = cv_df.as_matrix()[:,0]
# # 输出准确率
# print metrics.accuracy_score(predictions,y)


## 4.1.3 model系数和feature关系
# print pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})


## 4.1.4 判断过拟合/欠拟合
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
#     plt.figure(figsize=(10,6))
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel(scoring)
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# plot_learning_curve(clf, 'Logistic Regression', X, y, cv=4)
# plt.show()


## 4.2 SVM

### 4.2.1 交叉训练
# from sklearn import svm

# split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# svm_classifer = svm.SVC(kernel='rbf', gamma=0.7)
# svm_classifer.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = svm_classifer.predict(cv_df.as_matrix()[:,1:])
# y = cv_df.as_matrix()[:,0]
# # # 输出准确率
# print metrics.accuracy_score(predictions,y)


## 4.3 RandomForestClassifier

### 4.3.1 交叉训练
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(criterion='gini', 
#                              n_estimators=700,
#                              min_samples_split=10,
#                              min_samples_leaf=1,
#                              max_features='auto',
#                              oob_score=True,
#                              random_state=1,
#                              n_jobs=-1)

# split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# rf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = rf.predict(cv_df.as_matrix()[:,1:])
# y = cv_df.as_matrix()[:,0]

# print metrics.accuracy_score(predictions,y)

## 4.4 KNN

### 4.4.1 交叉训练
# from sklearn import neighbors 
# from sklearn.neighbors import KNeighborsClassifier

# split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

# y = cv_df.as_matrix()[:,0]
# knn = KNeighborsClassifier(n_neighbors=23)
# knn.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
# y_pred = knn.predict(cv_df.as_matrix()[:,1:])

# print metrics.accuracy_score(y_pred,y)


## 4.5 DecisionTree

### 4.5.1 交叉训练
# from sklearn.tree import DecisionTreeClassifier

# split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# y = cv_df.as_matrix()[:,0]

# tree = DecisionTreeClassifier()
# tree.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
# y_pred = tree.predict(cv_df.as_matrix()[:,1:])
# print metrics.accuracy_score(y_pred,y)




# # 5 处理test数据
# data_test = pd.read_csv('test.csv')
# data_test.loc[(data_test.Fare.isnull()),'Fare']=0
# ## 年龄处理
# tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# X = null_age[:,1:]
# predictiedAges = rfr.predict(X)
# data_test.loc[(data_test.Age.isnull()),'Age'] = predictiedAges
# ## Cabin处理
# data_test = set_Cabin_type(data_test)
# ## 对类目性特征因子化
# dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
# dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
# dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
# dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')
# df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
# df_test.drop(['Pclass','Sex','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)
# ## 归一化
# df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'],age_scale_param)
# df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'],fare_scale_param)

# # 6 预测
# test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions= clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
# result.to_csv('predictions_KNN.csv',index=False)






