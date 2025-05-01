import pandas as pd
import numpy as np
from pandas import factorize

#For Logistic Regression Implementation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_excel(r'C:\Users\rathi\OneDrive\Desktop\Extra\Python_work\marketing_campaign_wines.xlsx',
                   sheet_name='data')

print(df.head())

# Create Age Column in dataset using Year_Birth
df['Age'] = 2022 - df['Year_Birth']


for col in df.columns:
    print(col)

#df.describe()
df['MntWines'].describe()

#Plotting Histogram for Wine Spend
bin_values = np.arange(start=0, stop=2000, step=50)
df['MntWines'].hist(bins=bin_values, figsize=[14, 6])

#Plotting Wine Spend with other Numeric Variables and Categorical Variables and checking their Correlation
df.plot.scatter(x='MntWines', y='Income', marker='o', figsize=(7, 5))
df.plot.scatter(x='MntWines', y='MntFruits', marker='o', figsize=(7, 5))
df.plot.scatter(x='MntWines', y='NumStorePurchases', marker='o', figsize=(7, 5))

df['MntWines'].corr(df['Income']) #0.58
df['MntWines'].corr(df['MntMeatProducts']) #0.56
df['MntWines'].corr(df['NumDealsPurchases']) #
df['MntWines'].corr(df['NumWebPurchases']) #0.54
df['MntWines'].corr(df['NumCatalogPurchases']) #0.64
df['MntWines'].corr(df['NumStorePurchases']) #0.64
df['MntWines'].corr(df['Age']) #0.16

labels, categories = factorize(df["Marital_Status"])
df["labels"] = labels
abs(df["MntWines"].corr(df["labels"])) #0.07

labels2, categories2 = factorize(df["Education"])
df["labels2"] = labels2
abs(df["MntWines"].corr(df["labels2"]))

#Getting Full Correlation Matrix
corr_mat = df.corr(method='pearson')
upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
unique_corr_pairs = upper_corr_mat.unstack().dropna()
sorted_mat = unique_corr_pairs.sort_values()
print(sorted_mat)

#Getting Correlation Results on Excel
df2 = sorted_mat.to_frame()
df2.to_csv(r'C:\Users\rathi\OneDrive\Desktop\Extra\Python_work\corr_output.csv')


#Condition of "Wine Purchaser" based on 25th Percentile value (which is 23) on amount spent on wine
#This variable 'Wine_Purchaser' created would be used to differentiate between Wine Purchase or not
#We will use Logistic Regression with this variable as dependent variable

conditions = [
    (df['MntWines'] <= 25),
    (df['MntWines'] > 25)
    ]

values = ['0', '1']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Wine_Purchaser'] = np.select(conditions, values)

# display updated DataFrame
df.head()


#Logistic Regression - Implementation

#List of variables that have good correlation with Wine Spent
to_keep=['Kidhome', 'NumWebVisitsMonth', 'Response','AcceptedCmp1','AcceptedCmp4', 'MntSweetProducts',
         'MntGoldProds',
         'MntFruits','MntFishProducts','AcceptedCmp5','NumWebPurchases','MntMeatProducts','Income',
         'NumCatalogPurchases','NumStorePurchases','Wine_Purchaser']

data_final=df[to_keep]
data_final2= data_final.dropna()


pred_col=['Kidhome', 'NumWebVisitsMonth', 'Response','AcceptedCmp1','AcceptedCmp4', 'MntSweetProducts',
         'MntGoldProds',
         'MntFruits','MntFishProducts','AcceptedCmp5','NumWebPurchases','MntMeatProducts','Income',
         'NumCatalogPurchases','NumStorePurchases']

X=data_final2[pred_col]
y=data_final2['Wine_Purchaser']

# Creating the data sets for using in regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Predicting on the Test Sample
y_pred = logreg.predict(X_test)


#Evaluting the Performance of the Model
#Accuracy, Confusion Matrix, Classification Report

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))