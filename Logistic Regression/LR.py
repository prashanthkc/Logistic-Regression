####################################problem 1######################################]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#Importing Data
affairs = pd.read_csv("C:/Users/hp/Desktop/Logistic R assi/Affairs.csv", sep = ",")

#removing unwanted column
affairs = affairs.iloc[:,1:]
affairs.head(11)
affairs.describe()
affairs.columns
affairs.isna().sum()

#discretizing naffairs column , <6 as 0 and >=6 as 1
affairs['naffairs'] = (affairs['naffairs'] >= 6 ).astype(int)

# Model building 
logit_model = sm.logit('naffairs ~  kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 +yrsmarr5 + yrsmarr6',
                       data = affairs).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(affairs.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(affairs.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
affairs["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
affairs.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(affairs["pred"], affairs["naffairs"])
print(classification)


#Splitting the data into train and test data 
train_data, test_data = train_test_split(affairs, test_size = 0.3) # 30% test data

# Model building 
model = sm.logit('naffairs ~  kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 +yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[ :, 1: ])

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = accuracy_score(test_data['naffairs'], test_data.test_pred) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
print(classification_test)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = accuracy_score(train_data['naffairs'], train_data.train_pred) 
accuracy_train
############################################ Problem 2##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#Importing Data
ads = pd.read_csv("C:/Users/hp/Desktop/Logistic R assi/advertising.csv", sep = ",")

#removing unwanted columns
ads = ads.iloc[:,[0,1,2,3,6,9]]
ads.head(10)
ads.describe()
ads.columns
ads.isna().sum()

#renaming some columns
cols = {'Daily_Time_ Spent _on_Site':'Daily_Time_Spent_on_Site' ,'Daily Internet Usage':'Daily_Internet_Usage' }
ads.rename(cols , axis =1 , inplace = True)
# Model building 
logit_model = sm.logit('Clicked_on_Ad ~  Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male',
                       data = ads).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(ads.iloc[ :, :5])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(ads.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
ads["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
ads.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(ads["pred"], ads["Clicked_on_Ad"])
print(classification)

#Splitting the data into train and test data 
train_data, test_data = train_test_split(ads, test_size = 0.3) # 30% test data

# Model building 
model = sm.logit('Clicked_on_Ad ~  Daily_Time_Spent_on_Site + Age + Area_Income + Daily_Internet_Usage + Male', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[ :, :5])

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Clicked_on_Ad'])
confusion_matrix

accuracy_test = accuracy_score(test_data['Clicked_on_Ad'], test_data.test_pred) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Clicked_on_Ad"])
print(classification_test)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Clicked_on_Ad"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :5])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Clicked_on_Ad'])
confusion_matrx

accuracy_train = accuracy_score(train_data['Clicked_on_Ad'], train_data.train_pred) 
accuracy_train

############################################Problem 3####################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#Importing Data
election_data = pd.read_csv("C:/Users/hp/Desktop/Logistic R assi/election_data.csv", sep = ",")

#removing unwanted columns and row
election_data = election_data.iloc[1:,1:]
election_data.head(10)
election_data.describe()
election_data.columns
election_data.isna().sum()

#renaming some columns
cols = {'Amount Spent':'Amount_Spent' ,'Popularity Rank':'Popularity_Rank' }
election_data.rename(cols , axis =1 , inplace = True)
# Model building 
logit = sm.logit('Result ~  Year + Amount_Spent + Popularity_Rank',data = election_data)
logit_model = logit.fit(method = 'bfgs')
#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(election_data.iloc[ :, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(election_data.Result, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
election_data["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
election_data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(election_data["pred"], election_data["Result"])
print(classification)

#Splitting the data into train and test data 
train_data, test_data = train_test_split(election_data, test_size = 0.3) # 30% test data

# Model building 
final_logit = sm.logit('Result ~  Year + Amount_Spent + Popularity_Rank',data = train_data)
model = logit.fit(method = 'bfgs')
#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[ :, 1:])

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = accuracy_score(test_data['Result'], test_data.test_pred) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result"])
print(classification_test)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1:])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = accuracy_score(train_data['Result'], train_data.train_pred) 
accuracy_train

#####################################Problem 4########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#Importing Data
bank_data = pd.read_csv("C:/Users/hp/Desktop/Logistic R assi/bank_data.csv", sep = ",")

#removing unwanted columns
bank_data.head(10)
bank_data.describe()
bank_data.columns
bank_data.isna().sum()

cols_data = ['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign',
       'pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess',
       'poutunknown', 'con_cellular', 'con_telephone', 'con_unknown',
       'divorced', 'married', 'single', 'joadmin.', 'joblue.collar',
       'joentrepreneur', 'johousemaid', 'jomanagement', 'joretired',
       'joself.employed', 'joservices', 'jostudent', 'jotechnician',
       'jounemployed', 'jounknown']
# Model building 
logit_model = sm.Logit(bank_data['y'], bank_data[cols_data]).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()
pred = logit_model.predict(bank_data.iloc[ :, :31])
# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(bank_data.y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
bank_data["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
bank_data.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(bank_data["pred"], bank_data["y"])
print(classification)

#Splitting the data into train and test data 
train_data, test_data = train_test_split(bank_data, test_size = 0.3) # 30% test data

# Model building 
model = sm.Logit(train_data['y'], train_data[cols_data]).fit()
#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = model.predict(test_data.iloc[ :, :31])

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = accuracy_score(test_data['y'], test_data.test_pred) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
print(classification_test)

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :31])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = accuracy_score(train_data['y'], train_data.train_pred) 
accuracy_train
###################################################END##################################