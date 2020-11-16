import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB


pd.set_option("display.max_columns", 100)
pd.set_option('display.width', 1000)

path = 'F:/ML_Projects/BankChurners/BankChurners.csv'
df = pd.read_csv(path)


#-------------some pre data processing(uite obvious ones) to generate hypothesis-----------------------
df['Attrition_Flag'].replace({'Existing Customer':0, 'Attrited Customer':1},inplace=True)
df.drop(['CLIENTNUM'],axis=1,inplace=True)
df.drop(df.columns[[-1,-2]],axis=1,inplace=True)


#---------------hypothesis generation------------------
# existing = df.loc[df['Attrition_Flag']==0]
# nonExisting = df.loc[df['Attrition_Flag']==1]

# a = nonExisting['Card_Category'].value_counts()
# b = existing['Card_Category'].value_counts()
# cards = a.index.values
# differncePerc = []
# for c in cards:
#     print(c, end=' ')
#     print(a[c]/(a[c]+b[c]))
#     differncePerc.append(a[c]/(a[c]+b[c]))
#
# plt.bar(cards,differncePerc)
# plt.show()

# plt.hist(existing['Card_Category'],color='red',label='Existing')
# plt.hist(nonExisting['Card_Category'],color='blue',align='right',label='Non Existing')
# plt.legend(loc='upper right')
# plt.show()

# a = nonExisting['Income_Category'].value_counts()
# b = existing['Income_Category'].value_counts()
# incomes = a.index.values
# differncePerc = []
# for i in incomes:
#     print(i, end=' ')
#     print(a[i]/(a[i]+b[i]))
#     differncePerc.append(a[i]/(a[i]+b[i]))
#
# plt.bar(incomes,differncePerc)
# plt.show()


#--------------let's convert some categorical variables into numerical--------------
# catego_data = df.select_dtypes(exclude=[np.number])
# print(catego_data.head(10))
# print(catego_data['Education_Level'].unique())
# print(catego_data['Income_Category'].unique())
# print(catego_data['Card_Category'].unique())

#ordinal to numerical
map_education_level = {'High School':1,'Graduate':3,'Uneducated':0,'College':2,'Post-Graduate':4,'Doctorate':5}
map_income_level = {'$60K - $80K':3,'Less than $40K':1, '$80K - $120K':4,'$40K - $60K':2,'$120K +':5}
map_card_category = {'Blue':1,'Gold':3,'Silver':2,'Platinum':4}
df['Education_Level'].replace(map_education_level,inplace=True)
df['Income_Category'].replace(map_income_level,inplace=True)
df['Card_Category'].replace(map_card_category,inplace=True)

#hot encoding of gender category
df.insert(2,'Gender_M',df['Gender'],True)
df.rename({'Gender':'Gender_F'},axis=1,inplace=True)
df['Gender_M'].replace({'M':1,'F':0},inplace=True)
df['Gender_F'].replace({'M':0,'F':1},inplace=True)

#hot encoding of marital status
df.insert(7,'Single',df['Marital_Status'],True)
df.insert(7,'Divorced',df['Marital_Status'],True)
df.insert(7,'Unknown',df['Marital_Status'],True)
df.rename({'Marital_Status':'Married'},axis=1,inplace=True)
df['Married'].replace({'Single':0, 'Married':1, 'Divorced':0, 'Unknown':0},inplace=True)
df['Single'].replace({'Single':1, 'Married':0, 'Divorced':0, 'Unknown':0},inplace=True)
df['Divorced'].replace({'Single':0, 'Married':0, 'Divorced':1, 'Unknown':0},inplace=True)
df['Unknown'].replace({'Single':0, 'Married':0, 'Divorced':0, 'Unknown':1},inplace=True)

#----------------------dealing with mising values--------------------
# print('Income_Category')
# print(df['Income_Category'].value_counts()['Unknown']/df.shape[0]*100)
# print('Education_Level')
# print(df['Education_Level'].value_counts()['Unknown']/df.shape[0]*100)
# print('Unknown')
# print(df['Unknown'].value_counts()[1]/df.shape[0]*100)


#---------------------since number of missing values is very high, we are treating them as another value
educatedDF = df.loc[df['Education_Level']!='Unknown']
# print(educatedDF['Education_Level'].skew())
# sbn.displot(educatedDF,x='Education_Level')
# plt.show()
# print(educatedDF['Education_Level'].mean())
# print(educatedDF['Education_Level'].median())
# print(educatedDF['Education_Level'].mode())
mean_education = educatedDF['Education_Level'].mean()
df['Education_Level'].replace({'Unknown':mean_education},inplace=True)


salariedDF = df.loc[df['Income_Category']!='Unknown']
# sbn.displot(salariedDF,x='Income_Category')
# plt.show()
# print(salariedDF['Income_Category'].mean())
# print(salariedDF['Income_Category'].median())
# print(salariedDF['Income_Category'].mode())
# print(salariedDF['Income_Category'].skew())
median_salaries = salariedDF['Income_Category'].median()
df['Income_Category'].replace({'Unknown':median_salaries},inplace=True)

# index_unknown_education = df.index[df['Education_Level']=='Unknown'].tolist()
# df.insert(5,'Unknown Education',0,True)
# df.loc[index_unknown_education,'Unknown Education'] = 1

# print(df.describe())
# print(df.head(15))
# print(df[df.eq('Unknown').any(1)])   ----- confirming there's no missing values anymore


#-------dataset split-----------
x = df.iloc[:,1:]
y = df.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)


#---------------------------fit into model-------------------------

#____Logistic Regression______
# lr = LogisticRegression(max_iter=2000)
# lr.fit(x_train,y_train)
# y_predict = lr.predict(x_test)
# print(confusion_matrix(y_test,y_predict))
# print(classification_report(y_test,y_predict))

#__________Naive Bayes______________
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(x_test)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
