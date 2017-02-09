# Titanic

## 目录
-  一、库
-  二、数据认识
	- 导入数据
	- 数据初认识
	- 数据图像展示
-  三、数据预处理
	- 清洗无用数据
	- 补全数据
	- 对类目性特征因子化
	- 部分数据归一化
-  四、机器学习模型
    - 逻辑回归
    - KNN
    - SVM
    - RandomForestClassifier
    - DecisionTree
-  五、处理test数据
-  六、预测

## 一、库
- 导入相关库
```python
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
```

## 二、数据认识

### 2.1 导入数据
```python
titanic_data = pd.read_csv('train.csv')
```
### 2.2 数据初认识
- 数据基本信息
```python
print titanic_data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None
```
数据共有11个特征，1个标识(Survived)，其中Age,Cabin,Embarked有缺省值。

- 数据展示
```python
print titanic_data.head()

   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S 
```
### 2.2 数据图像展示
- 乘客属性的分布
```python
fig = plt.figure(figsize=(20,10))
fig.set(alpha=0.2)

plt.subplot2grid((3,3),(0,0))
titanic_data.Survived.value_counts().plot(kind='bar')
plt.title(u"获救情况(1为获救)")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(0,1))
titanic_data.Pclass.value_counts().plot(kind='bar')
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(0,2))
plt.scatter(titanic_data.Survived,titanic_data.Age)
plt.grid(b=True,which='major',axis='y')
plt.title(u"按年龄看获救分布")
plt.ylabel(u"年龄")

plt.subplot2grid((3,3),(1,0),colspan=2)
titanic_data.Age[titanic_data.Pclass == 1].plot(kind='kde')
titanic_data.Age[titanic_data.Pclass == 2].plot(kind='kde')
titanic_data.Age[titanic_data.Pclass == 3].plot(kind='kde')
# plt.xlabel(u'年龄')
plt.ylabel(u'密度')
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱',u'2等仓',u'3等仓'),loc='best')

plt.subplot2grid((3,3),(1,2))
titanic_data.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

plt.subplot2grid((3,3),(2,0),colspan=2)
titanic_data.Age[titanic_data.Survived == 1].plot(kind='kde')
titanic_data.Age[titanic_data.Survived == 0].plot(kind='kde')
plt.xlabel(u'年龄')
plt.ylabel(u'密度')
plt.title(u"获救与没获救乘客年龄分布")
plt.legend((u'获救',u'没获救'),loc='best')

plt.show()
```
![](raw/figure_1.png?raw=true)

- 乘客属性与获救结果的关联统计
 - 乘客舱位等级的获救情况
 ![](raw/figure_2.png?raw=true)
 
 - 乘客性别的获救情况
 ![](raw/figure_3.png?raw=true)
 
 - 各种舱级别情况下各性别的获救情况
 ![](raw/figure_4.png?raw=true)
 
 - 各登船港口的获救情况
 ![](raw/figure_5.png?raw=true)
 
## 三、数据预处理

### 3.1 清洗无用数据
```python 
titanic_data = titanic_data.drop(['PassengerId','Name','Ticket'],axis=1)
```

### 3.2 补全数据
- Age字段 采用RandomForestRegressor自动补全
```python
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
```

- Cabin字段 简单处理成有无
```python
def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
	df.loc[(df.Cabin.isnull()),'Cabin'] = "No"
	return df
```

- Embarked字段 填充为C港口登陆
```python
def set_Embarked_type(df):
	df['Embarked'] = df['Embarked'].fillna('C')
	return df
```

### 3.3 对类目性特征因子化
```python
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df.drop(['Pclass','Sex','Cabin','Embarked'],axis=1,inplace=True)
```

### 3.4 部分数据归一化
```python
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale_param)

```

## 四、机器学习模型

### 4.1 逻辑回归
- 交叉训练
```python
split_train,split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

clf = linear_model.LogisticRegression(C=1.0,tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])
y = cv_df.as_matrix()[:,0]
# 输出准确率
print metrics.accuracy_score(predictions,y)
```
输出结果: 0.817164179104

- model系数和feature关系
```
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:,0]
X = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0,tol=1e-6)
clf.fit(X,y)

print pd.DataFrame({'columns':list(train_df.columns)[1:],'coef':list(clf.coef_.T)})


                 coef      columns
0   [-0.348497390212]        SibSp
1   [-0.115011572237]        Parch
2   [-0.336338628797]     Cabin_No
3     [0.59167626797]    Cabin_Yes
4    [0.239978778481]   Embarked_C
5    [0.219015266472]   Embarked_Q
6    [-0.20365640578]   Embarked_S
7     [1.44257508282]   Sex_female
8    [-1.18723744365]     Sex_male
9    [0.703203018111]     Pclass_1
10   [0.369457932227]     Pclass_2
11  [-0.817323311164]     Pclass_3
12  [-0.532966826446]   Age_scaled
13  [0.0919778701395]  Fare_scaled
```

- 判断过拟合/欠拟合
```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

plot_learning_curve(clf, 'Logistic Regression', X, y, cv=4)
plt.show()
```

![](raw/figure_6.png?raw=true)

## 五、处理test数据 
- 与第三步处理过程类似
```python
data_test = pd.read_csv('test.csv')
data_test.loc[(data_test.Fare.isnull()),'Fare']=0
## 年龄处理
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:,1:]
predictiedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictiedAges
## Cabin处理
data_test = set_Cabin_type(data_test)
## 对类目性特征因子化
dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')
df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
df_test.drop(['Pclass','Sex','Cabin','Embarked','Name','Ticket'],axis=1,inplace=True)
## 归一化
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'],age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'],fare_scale_param)
```

## 六、预测
- 把预测结果写入到predictions.csv中
```python
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions= clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('predictions.csv',index=False)
```

- Kaggle递交结果
![](raw/figure_7.png?raw=true)
