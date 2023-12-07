#!/usr/bin/env python
# coding: utf-8

# In[713]:


import pandas as pd


# In[714]:


train=pd.read_csv(r"C:\Users\Amol purkar\Desktop\HR_analytics-Employee_Promotion\train.csv")
train


# In[715]:


test=pd.read_csv(r"C:\Users\Amol purkar\Desktop\HR_analytics-Employee_Promotion\test.csv")
test


# In[716]:


print("In Train Data there is ",train.shape[0]," Rows and ", train.shape[1], "Columns" )


# In[717]:


print("In Test Data there is ",test.shape[0]," Rows and ", test.shape[1], "Columns" )


# In[718]:


train.columns


# In[719]:


test.columns


# employee_id: Unique ID for employee.
# 
# department: Department of employee.
# 
# region: Region of employment (unordered).
# 
# education: Education Level.
# 
# gender: Gender of Employee.
# 
# recruitment_channel: Channel of recruitment for employee.
# 
# no_of_trainings: no of other trainings completed in previous year on soft skills, technical skills etc.
# 
# age: Age of Employee
# 
# previous_ year_rating: Employee Rating for the previous year.
# 
# length_ of_ service: Length of service in years.
# 
# awards_ won?: if awards won during previous year then 1 else 0.
# 
# avg_ training_ score: Average score in current training evaluations.
# 
# is_promoted: (Target) Recommended for promotion.

# In[720]:


train.describe()


# In[721]:


#descriptive statististics for categorical columns
train.describe(include = 'object')


# In[722]:


train.info()


# In[723]:


#Number of Unique Values
unique_values= train.select_dtypes(include='number').nunique()


# In[ ]:





# In[724]:


import matplotlib.pyplot as plt


# In[725]:


train.columns


# In[726]:


train.is_promoted.value_counts()


# In[727]:


# importing packages
import seaborn as sns
import matplotlib.pyplot as plt


sns.countplot(x='is_promoted', data=train, )
plt.show()


# In[728]:


Null=train.isnull().sum() [train.isnull().sum()>0]
Null


# There is null values in two table Education and Previous Year Rating 

# In[729]:


train.education.value_counts()


# In[730]:


train.education=train.education.fillna("Bachelor's")


# In[731]:


train.previous_year_rating.value_counts()


# In[732]:


train.previous_year_rating=train.previous_year_rating.fillna(3.0)


# Now all null values filled .There is no any null values.

# In[733]:


train.gender.value_counts()


# In[734]:


plt.figure(figsize=(5,5))
sns.countplot(x=train.gender)
plt.show()


# The ratio of Male Employees higher than the Female Employees

# In[735]:


train.department.value_counts()


# In[736]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.department,order=train.department.value_counts().index)
plt.xlabel("Departments")
plt.show()


# In[737]:


train.region.value_counts()


# In[738]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.region,order=train.region.value_counts().index)
plt.xticks(rotation=90)
plt.show()


# Most of the employees is from region 2

# In[739]:


plt.figure(figsize=(5,5))
sns.countplot(x=train.education)
plt.show()


# We saw most of employees there Bachelor's Degree

# In[740]:


#Number of training gone through
plt.figure(figsize=(15,5))
sns.countplot(x=train.no_of_trainings)
plt.show()


# In[741]:


#age distributions
plt.figure(figsize=(15,5))
sns.countplot(x=train.age)
plt.show()


# Most of Employees age is bet 29 to 33

# In[742]:


#previoues year rating
plt.figure(figsize=(15,5))
sns.countplot(x=train.previous_year_rating)
plt.show()


# In[743]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.recruitment_channel)
plt.show()


# Most of employees of Recruitment channel is other

# In[744]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.length_of_service)
plt.show()


# In[745]:


#Most of employee have worked for 1 to 7 years at the firm


# In[746]:


train=train.rename(columns={"awards_won?":"awards_won"})
test=test.rename(columns={"awards_won?":"awards_won"})


# In[747]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.awards_won)
plt.show()


# In[748]:


(train.awards_won.value_counts()[1]/len(train))*100


# In[749]:


print("The percentage of won awards is ",(train.awards_won.value_counts()[1]/len(train))*100,"%")


# In[750]:


plt.figure(figsize=(20,20))
sns.countplot(x=train.avg_training_score)
plt.show()


# In[751]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.is_promoted)
plt.show()


# In[752]:


(train.is_promoted.value_counts()[1]/len(train))*100


# In[753]:


print("Percentage of employee who were promoted in previous year ",(train.is_promoted.value_counts()[1]/len(train))*100,"%")


# In[ ]:





# In[754]:


plt.figure(figsize=(15,5))
sns.countplot(x=train.department,hue=train.is_promoted,order=train.department.value_counts().index)
plt.show()


# In[755]:


#sales and Marketing ,Operation ,technology and Procurement is slightly high chance to get promoted


# In[756]:


# relationship bet gender and promotion


# In[757]:


plt.figure(figsize=(5,5))
sns.countplot(x=train.gender,hue=train.is_promoted,order=train.gender.value_counts().index)


# In[758]:


#Male has gether chance to promote


# In[759]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.region,order=train.region.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[760]:


#highest employees from region 2 so chances of promotion is also high


# In[761]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.education,order=train.education.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[762]:


#bachelor's have high chance to get promotion


# In[763]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.recruitment_channel,order=train.recruitment_channel.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[764]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.no_of_trainings,order=train.no_of_trainings.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[765]:


# get promoted which have 1 and 2 skills


# In[766]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.age,order=train.age.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[767]:


# age is 27 to 32 is more proting rather others


# In[768]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.previous_year_rating,order=train.previous_year_rating.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[769]:


# rating of 3 to 5 has high chances of Promotion


# In[770]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.length_of_service,order=train.length_of_service.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[771]:


# length of service does not effect on Promotion


# In[772]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.avg_training_score,order=train.avg_training_score.value_counts().index,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[773]:


plt.figure(figsize=(20,5))
sns.boxplot(x=train.department,y=train.avg_training_score,hue=train.is_promoted)
plt.xticks(rotation=90)
plt.show()


# In[774]:


#48 to 51 ,58 to 61,68 to 71 and 80 to 83 was promoted


# In[775]:


plt.figure(figsize=(20,5))
sns.boxplot(x=train.department,y=train.avg_training_score)

plt.show()


# In[776]:


#We see Hr , sales and marketing has lest score 

#tech ,Analytics And R&D departments has high score in training


# In[777]:


plt.figure(figsize=(20,5))
sns.countplot(x=train.awards_won,hue=train.is_promoted)

plt.show()


# In[778]:


# A employees who won awards had sure got a promotion


# In[779]:


plt.figure(figsize=(10,5))
sns.heatmap(train.corr(),annot=True)
plt.show()


# #There is posotive Correlation Between award won and promotion
# #There is mild positive correlation between training and promotion, showing the employees who got higher score in training get promoted
# 
# #there is mild positive correlation between previous year rating and promotion ,there is high chances of getting promotion who has higher rating in previous year
# #there is strong positive corrlation between length of service and promotion.

# # DATA Preprocessing

# In[780]:


from sklearn.preprocessing import LabelEncoder


# In[781]:


le=LabelEncoder()


# In[782]:


test.info()


# In[783]:


test.department=le.fit_transform(test.department)
test.region=le.fit_transform(test.region)
test.education=le.fit_transform(test.education)
test.gender=le.fit_transform(test.gender)
test.recruitment_channel=le.fit_transform(test.recruitment_channel)


# In[784]:


train.department=le.fit_transform(train.department)
train.region=le.fit_transform(train.region)
train.education=le.fit_transform(train.education)
train.gender=le.fit_transform(train.gender)
train.recruitment_channel=le.fit_transform(train.recruitment_channel)
train.no_of_trainings=le.fit_transform(train.no_of_trainings)


# In[ ]:





# In[840]:


from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


# In[ ]:





# In[ ]:





# In[841]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(train_x, train_y)


# In[842]:


pred_log = log.predict(test_x)


# In[843]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[846]:


tab_ch = confusion_matrix(test_y, pred_log)
tab_ch


# In[855]:


accuracy = accuracy_score(test_y, pred_log)
print("Accuracy",accuracy)
precision = precision_score(test_y, pred_log)
print("precision",precision)
recall = recall_score(test_y, pred_log)
print("recall",recall)
f1 = f1_score(test_y, pred_log)
print("F1 score ",f1)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[849]:


roc_auc = roc_auc_score(test_y, log.predict_proba(test_x)[:, 1])
roc_auc


# In[ ]:





# In[859]:


from sklearn.ensemble import RandomForestClassifier

# Try a different algorithm that might capture the positive class better
rd = RandomForestClassifier(n_estimators=100)
rd.fit(train_x, train_y)


# In[860]:


pred_rd = rd.predict(test_x)


# In[861]:


tab_rd = confusion_matrix(test_y, pred_rd)
tab_rd


# In[862]:


accuracy = accuracy_score(test_y, pred_rd)
print("Accuracy",accuracy)
precision = precision_score(test_y, pred_rd)
print("precision",precision)
recall = recall_score(test_y, pred_rd)
print("recall",recall)
f1 = f1_score(test_y, pred_rd)
print("F1 score ",f1)


# In[ ]:




