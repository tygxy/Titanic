#coding:utf-8
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 0.导入数据
titanic_df = pd.read_csv("train.csv")
test_df = pd.read_csv('test.csv')

# 1.数据初览
# print titanic_df.head()
# print titanic_df.info()

# 2.数据预处理

## 2.1 去除无用特征
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket'], axis = 1)


## 2.2 特征工程

### 2.2.1 Embarked
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
# sns.factorplot('Embarked', 'Survived', data = titanic_df, size = 4, aspect = 3)
# 绘制三幅图
# fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
# sns.countplot(x = 'Embarked', data = titanic_df, ax = axis1)
# sns.countplot(x = 'Survived', hue = 'Embarked', data = titanic_df, order = [1, 0], ax = axis2) 
# embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index = False).mean()
# sns.barplot(x = 'Embarked', y = 'Survived', data = embark_perc, order = ['S', 'C', 'Q'], ax = axis3)
# 第三幅图等价于
# print titanic_df['Survived'].groupby(titanic_df['Embarked']).mean()
# plt.show()

embark_dummies_titanic = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis = 1, inplace = True)

embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis = 1, inplace = True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis = 1, inplace = True)
test_df.drop(['Embarked'], axis = 1, inplace = True)


### 2.2.2 Fare
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)

# 绘图
# fare_not_survived = titanic_df['Fare'][titanic_df['Survived'] == 0]
# fare_survived = titanic_df['Fare'][titanic_df['Survived'] == 1]

# avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

# titanic_df['Fare'].plot(kind = 'hist', figsize = (15, 3), bins = 100, xlim = (0, 50))
# avgerage_fare.index.names = std_fare.index.names = ['Survived']
# avgerage_fare.plot(yerr = std_fare, kind = 'bar', legend = False)

# plt.show()



### 2.2.3 Age
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (15, 4))
# axis1.set_title('Original Age values - Titanic')
# axis2.set_title('New Age values - Titanic')

# get average, std, and number of NaN values in titanic_df
avgerage_age_titanic = titanic_df['Age'].mean()
std_age_titanic = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# 随机制造count_nan_age_test个年龄
rand_1 = np.random.randint(avgerage_age_titanic - std_age_titanic, avgerage_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# titanic_df['Age'].dropna().astype(int).hist(bins = 70, ax = axis1)

titanic_df['Age'][np.isnan(titanic_df['Age'])] = rand_1
test_df['Age'][np.isnan(test_df['Age'])] = rand_2

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)


# # 作图
# # titanic_df['Age'].hist(bins = 70, ax = axis2)

# # 继续作图
# # peaks for survived/not survived passengers by their age
# facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
# facet.map(sns.kdeplot,'Age',shade= True)
# facet.set(xlim=(0, titanic_df['Age'].max()))
# facet.add_legend()

# # average survived passengers by age
# # 每个年龄的存活率：
# fig, axis1 = plt.subplots(1,1,figsize=(18,4))
# average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
# sns.barplot(x='Age', y='Survived', data=average_age)

# plt.show()

### 2.2.4 Cabin
titanic_df.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)

### 2.2.5 Family
titanic_df['Family'] = titanic_df['Parch'] + titanic_df['SibSp']
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

titanic_df = titanic_df.drop(['Parch', 'SibSp'], axis = 1)
test_df = test_df.drop(['Parch', 'SibSp'], axis = 1)

# 作图
# fig, (axis1, axis2) = plt.subplots(1, 2, sharex = True, figsize = (10, 5))
# sns.countplot(x = 'Family', data =titanic_df, order = [1, 0], ax = axis1)

# family_perc = titanic_df[["Family","Survived"]].groupby(['Family'], as_index = False).mean()
# sns.barplot(x = 'Family', y = 'Survived', data = family_perc, order = [1, 0], ax = axis2)

# axis1.set_xticklabels(["with Family", "Alone"], rotation = 0)

# plt.show()

### 2.2.5 Sex 
def get_person(passenger):
	age, sex = passenger
	return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis = 1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis = 1)

titanic_df.drop(['Sex'], axis = 1, inplace = True)
test_df.drop(['Sex'], axis = 1, inplace = True)

person_dummies_titanic = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child', 'Female', 'Male']
person_dummies_titanic.drop(['Male'], axis = 1, inplace = True)

person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Female', 'Male']
person_dummies_test.drop(['Male'], axis = 1, inplace = True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df = test_df.join(person_dummies_test)

# 作图
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))
# sns.countplot(x = 'Person', data = titanic_df, ax = axis1)

# person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'], as_index = False).mean()
# sns.barplot(x = 'Person', y = 'Survived', data = person_perc , ax = axis2, order = ['male', 'female', 'child'])

# plt.show()

titanic_df.drop(['Person'], axis = 1, inplace = True)
test_df.drop(['Person'], axis = 1, inplace = True)

### 2.2.6 Pclass
# sns.factorplot('Pclass', 'Survived', order = [1, 2, 3], data = titanic_df, size = 5)
# plt.show()

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

# 3.训练模型
X_train = titanic_df.drop('Survived', axis = 1)
Y_train = titanic_df['Survived']
X_test = test_df.drop('PassengerId', axis = 1).copy()

### 3.1 Logistic Regression
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# logreg.score(X_train, Y_train)

### 3.2  Random Forests
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)


# 4.相关分析
# coeff_df = DataFrame(titanic_df.columns.delete(0))
# coeff_df.columns = ['Features']
# coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
# print coeff_df

# 5.提交预测结果
submission = pd.DataFrame({
	"PassengerId": test_df['PassengerId'],
	"Survived": Y_pred
	})
submission.to_csv('Prediction_V3_RF_20170317.csv', index = False)
















