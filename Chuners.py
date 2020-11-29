import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


pd.set_option("display.max_columns", 100)
pd.set_option('display.width', 1000)

path = 'F:/ML_Projects/BankChurners/BankChurners.csv'
df = pd.read_csv(path)


#-------------some pre data processing(quite obvious ones) to generate hypothesis-----------------------
df['Attrition_Flag'].replace({'Existing Customer':0, 'Attrited Customer':1},inplace=True)
df.drop(df.columns[[0,-1,-2]].values,axis=1,inplace=True)
print(df.head(2))
print(df.shape)



# #---------------hypothesis generation------------------
# # existing = df.loc[df['Attrition_Flag']==0]
# # nonExisting = df.loc[df['Attrition_Flag']==1]
#
# # a = nonExisting['Card_Category'].value_counts()
# # b = existing['Card_Category'].value_counts()
# # cards = a.index.values
# # differncePerc = []
# # for c in cards:
# #     print(c, end=' ')
# #     print(a[c]/(a[c]+b[c]))
# #     differncePerc.append(a[c]/(a[c]+b[c]))
# #
# # plt.bar(cards,differncePerc)
# # plt.show()
#
# # plt.hist(existing['Card_Category'],color='red',label='Existing')
# # plt.hist(nonExisting['Card_Category'],color='blue',align='right',label='Non Existing')
# # plt.legend(loc='upper right')
# # plt.show()
#
# # a = nonExisting['Income_Category'].value_counts()
# # b = existing['Income_Category'].value_counts()
# # incomes = a.index.values
# # differncePerc = []
# # for i in incomes:
# #     print(i, end=' ')
# #     print(a[i]/(a[i]+b[i]))
# #     differncePerc.append(a[i]/(a[i]+b[i]))
# #
# # plt.bar(incomes,differncePerc)
# # plt.show()


#------------Exploratory Data Analysis---------------------
# plt.hist(df['Customer_Age'],bins=40,density=True)    # age follows normal curve
# plt.xticks(range(25,75,1))
# plt.show()

# sizes = (df['Attrition_Flag'].value_counts()).tolist()
# plt.pie(sizes,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# plt.show()

#______gender based division
# sizes_f = df.loc[df['Gender']=='F']['Attrition_Flag'].value_counts()
# sizes_m = df.loc[df['Gender']=='M']['Attrition_Flag'].value_counts()
# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.pie(sizes_f,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# ax1.title.set_text('Females')
# ax2.pie(sizes_m,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# ax2.title.set_text('Males')
# plt.show()

# sizes_gender = df['Gender'].value_counts()    #not much difference. thus it's eually distributed.
# plt.pie(sizes_gender.tolist(),autopct='%1.2f%%',labels=sizes_gender.index.values,explode=[0.1,0],shadow=True)
# plt.show()

#______card category based
# sizes_b = df.loc[df['Card_Category']=='Blue']['Attrition_Flag'].value_counts().tolist()
# sizes_s = df.loc[df['Card_Category']=='Silver']['Attrition_Flag'].value_counts().tolist()
# sizes_g = df.loc[df['Card_Category']=='Gold']['Attrition_Flag'].value_counts().tolist()
# sizes_p = df.loc[df['Card_Category']=='Platinum']['Attrition_Flag'].value_counts().tolist()
# fig,((axs0, axs1), (axs2, axs3)) = plt.subplots(2,2)
# axs0.pie(sizes_b,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs0.title.set_text('Blue Card')
# axs1.pie(sizes_s,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs1.title.set_text('Silver Card')
# axs2.pie(sizes_g,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs2.title.set_text('Gold Card')
# axs3.pie(sizes_p,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs3.title.set_text('Platinum Card')
# plt.show()


#______income category based

# sizes_0 = df.loc[df['Income_Category']=='Less than $40K']['Attrition_Flag'].value_counts().tolist()
# sizes_40 = df.loc[df['Income_Category']=='$40K - $60K']['Attrition_Flag'].value_counts().tolist()
# sizes_60 = df.loc[df['Income_Category']=='$60K - $80K']['Attrition_Flag'].value_counts().tolist()
# sizes_80 = df.loc[df['Income_Category']=='$80K - $120K']['Attrition_Flag'].value_counts().tolist()
# sizes_120 = df.loc[df['Income_Category']=='$120K +']['Attrition_Flag'].value_counts().tolist()
# sizes_unkn = df.loc[df['Income_Category']=='Unknown']['Attrition_Flag'].value_counts().tolist()
#
# fig,((axs0, axs1, axs2), (axs3, axs4, axs5)) = plt.subplots(2,3)
# axs0.pie(sizes_0,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs0.title.set_text('<40K')
# axs1.pie(sizes_40,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs1.title.set_text('40-60K')
# axs2.pie(sizes_60,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs2.title.set_text('60-80K')
# axs3.pie(sizes_80,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs3.title.set_text('80-120K')
# axs4.pie(sizes_120,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs4.title.set_text('>120K')
# axs5.pie(sizes_unkn,explode=[0,0.1],shadow=True,autopct='%1.2f%%',labels=['Existing','Churned'])
# axs5.title.set_text('Unknown')
# plt.show()

#-----education based----
# sizes_edu = df['Education_Level'].value_counts()
# plt.pie(sizes_edu.tolist(),labels=sizes_edu.index.values,autopct='%1.2f%%')
# plt.show()
#
# #-------------------correlation between numeric variables and target--------------
# # numeric_data = df.select_dtypes(include=[np.number])
# # corr_numeric = numeric_data.corr()
# # sbn.heatmap(corr_numeric,cmap="YlGnBu",annot=True)
# # plt.xticks(rotation=45)
# # plt.show()
#
#
# #------based on correlation------
# # df['Total_Trans_Amt'] = df['Total_Trans_Amt']/df['Total_Trans_Ct']
# # df.rename({'Total_Trans_Amt':'Total_Trans_Avg_Amt'},axis=1,inplace=True)
# df.drop(['Avg_Open_To_Buy'],axis=1,inplace=True)
#
#
# #--------------let's convert some categorical variables into numerical--------------
# # catego_data = df.select_dtypes(exclude=[np.number])
# # print(catego_data.head(10))
# # print(catego_data['Education_Level'].unique())
# # print(catego_data['Income_Category'].unique())
# # print(catego_data['Card_Category'].unique())
#
# #ordinal to numerical
map_education_level = {'High School':1,'Graduate':3,'Uneducated':0,'College':2,'Post-Graduate':4,'Doctorate':5}
map_income_level = {'$60K - $80K':3,'Less than $40K':1, '$80K - $120K':4,'$40K - $60K':2,'$120K +':5}
map_card_category = {'Blue':1,'Gold':3,'Silver':2,'Platinum':4}
df['Education_Level'].replace(map_education_level,inplace=True)
df['Income_Category'].replace(map_income_level,inplace=True)
df['Card_Category'].replace(map_card_category,inplace=True)

#
# #hot encoding of gender category
df.insert(2,'Gender_M',df['Gender'],True)
df.rename({'Gender':'Gender_F'},axis=1,inplace=True)
df['Gender_M'].replace({'M':1,'F':0},inplace=True)
df['Gender_F'].replace({'M':0,'F':1},inplace=True)
#
# #hot encoding of marital status
df.insert(7,'Single',df['Marital_Status'],True)
df.insert(7,'Divorced',df['Marital_Status'],True)
df.insert(7,'Unknown',df['Marital_Status'],True)
df.rename({'Marital_Status':'Married'},axis=1,inplace=True)
df['Married'].replace({'Single':0, 'Married':1, 'Divorced':0, 'Unknown':0},inplace=True)
df['Single'].replace({'Single':1, 'Married':0, 'Divorced':0, 'Unknown':0},inplace=True)
df['Divorced'].replace({'Single':0, 'Married':0, 'Divorced':1, 'Unknown':0},inplace=True)
df['Unknown'].replace({'Single':0, 'Married':0, 'Divorced':0, 'Unknown':1},inplace=True)

# plt.hist(df.loc[df['Income_Category']!='Unknown']['Income_Category'])   # income is rightly skewed. so central value is median
# plt.show()

# plt.hist(df.loc[df['Education_Level']!='Unknown']['Education_Level'])   # education is normally distributed. so central value is mean
# plt.show()

#
# #----------------------dealing with mising values--------------------
# # print('Income_Category')
# # print(df['Income_Category'].value_counts()['Unknown']/df.shape[0]*100)
# # print('Education_Level')
# # print(df['Education_Level'].value_counts()['Unknown']/df.shape[0]*100)
# # print('Unknown')
# # print(df['Unknown'].value_counts()[1]/df.shape[0]*100)
#
#
# #---------------------since number of missing values is very high, we are treating them as another value
educatedDF = df.loc[df['Education_Level']!='Unknown']
# # print(educatedDF['Education_Level'].skew())
# # sbn.displot(educatedDF,x='Education_Level')
# # plt.show()
# # print(educatedDF['Education_Level'].mean())
# # print(educatedDF['Education_Level'].median())
# # print(educatedDF['Education_Level'].mode())
# # df['Unknown_Edu'] = np.where(df['Education_Level']=='Unknown',1,0)
mean_education = educatedDF['Education_Level'].mean()
df['Education_Level'].replace({'Unknown':mean_education},inplace=True)
#
salariedDF = df.loc[df['Income_Category']!='Unknown']
# # sbn.displot(salariedDF,x='Income_Category')
# # plt.show()
# # print(salariedDF['Income_Category'].mean())
# # print(salariedDF['Income_Category'].median())
# # print(salariedDF['Income_Category'].mode())
# # print(salariedDF['Income_Category'].skew())
# # df['Unknown_Salary'] = np.where(df['Income_Category']=='Unknown',1,0)
median_salaries = salariedDF['Income_Category'].median()
df['Income_Category'].replace({'Unknown':median_salaries},inplace=True)
# # print(df.head(20))
#
#
# # index_unknown_education = df.index[df['Education_Level']=='Unknown'].tolist()
# # df.insert(5,'Unknown Education',0,True)
# # df.loc[index_unknown_education,'Unknown Education'] = 1
#
# # print(df.describe())
# # print(df.head(15))
# # print(df[df.eq('Unknown').any(1)])   ----- confirming there's no missing values anymore
#
# #-------dataset split-----------
x = df.iloc[:,1:]
y = df.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
#
#
# #---------------------------fit into model-------------------------
#
# #____Logistic Regression______
# # lr = LogisticRegression(max_iter=2000)
# # lr.fit(x_train,y_train)
# # y_predict = lr.predict(x_test)
# # print(confusion_matrix(y_test,y_predict))
# # print(classification_report(y_test,y_predict))
#
# #__________Naive Bayes______________
# # gnb = GaussianNB()
# # gnb.fit(x_train,y_train)
# # y_predict = gnb.predict(x_test)
# # print(confusion_matrix(y_test,y_predict))
# # print(classification_report(y_test,y_predict))
#
# #______________SVM________________
# svm = svm.SVC(kernel='linear')
# svm.fit(x_train,y_train)
# y_predict = svm.predict(x_test)
# print(confusion_matrix(y_test,y_predict))
# print(classification_report(y_test,y_predict))

# print(y_predict.sum())

#-----RANDOM FOREST-----
print(df.head(1))
#got recall of 74%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train[0])
#
# # Fitting Random Forest Classifier to the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, random_state = 0)
classifier.fit(x_train, y_train)
# Predicting result for training set and validation set
# predict_train_rf = classifier.predict(x_train)
# predict_val_rf = classifier.predict(x_test)
#
# from sklearn.metrics import confusion_matrix
# cm_train = confusion_matrix(y_train, predict_train_rf)
# cm_val = confusion_matrix(y_test, predict_val_rf)
#
# # Model Performance
from sklearn.metrics import accuracy_score, recall_score, precision_score
# print("Val Accuracy : ", accuracy_score(y_test, predict_val_rf) *  100)
# print("Val Recall : ", recall_score(y_test, predict_val_rf) *  100)
# print("Val Precision : ", precision_score(y_test, predict_val_rf) *  100)
# print(confusion_matrix(y_test, predict_val_rf))
# print(classification_report(y_test, predict_val_rf))


#-----Upsampling----
from sklearn.utils import resample
from collections import Counter

print("Before Upsampling:-")
print(Counter(y_train))

# X_train_upsampled, y_train_upsampled = resample(x_train[y_train == 1],
#                                                 y_train[y_train == 1],
#                                                 replace=True,
#                                                 n_samples=x_train[y_train == 0].shape[0],
#                                                 random_state=123)


# Let's use SMOTE to oversample
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train_upsampled, y_train_upsampled = oversample.fit_resample(x_train,y_train)

print("After Upsampling:-")
print(Counter(y_train_upsampled))

# X_train_bal = X_train_upsampled
# y_train_bal = y_train_upsampled


#-----Random Forest after upsampling------
print("\n\n\n\n AFTER UPSAMPLING\n\n")
classifier = RandomForestClassifier(n_estimators = 50, random_state = 0)
classifier.fit(X_train_upsampled, y_train_upsampled)
# Predicting result for training set and validation set
# predict_train_rf = classifier.predict(X_train_bal)
predict_val_rf = classifier.predict(x_test)


cm_val = confusion_matrix(y_test, predict_val_rf)

# Model Performance

print("Val Accuracy : ", accuracy_score(y_test, predict_val_rf) *  100)
print("Val Recall : ", recall_score(y_test, predict_val_rf) *  100)
print("Val Precision : ", precision_score(y_test, predict_val_rf) *  100)
print(cm_val)
print(classification_report(y_test, predict_val_rf))

#this gave us recall of 86%
