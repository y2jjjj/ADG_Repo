import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Fraud.csv")
print(data.shape)
print(data.head(200))
print(data.tail(200))
print(data.isnull().values.any())
print(data.info())

# Count legit and fraud transactions
ok_txns = len(data[data.isFraud == 0])
bad_txns = len(data[data.isFraud == 1])
ok_pct = (ok_txns / (bad_txns + ok_txns)) * 100
bad_pct = (bad_txns / (bad_txns + ok_txns)) * 100

print("Good transactions:", ok_txns)
print("Bad transactions:", bad_txns)
print("Good transactions %: {:.4f} %".format(ok_pct))
print("Bad transactions %: {:.4f} %".format(bad_pct))

# Look at merchant data
merch_data = data[data['nameDest'].str.contains('M')]
print(merch_data.head())

print(data.dtypes)

num_data = data.select_dtypes(include=[np.number])
corr_mat = num_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_mat, annot=True)
plt.show()

# Plot transaction types
plt.figure(figsize=(5,10))
labels = ["Good", "Bad"]
txn_counts = data.value_counts(data['isFraud'], sort=True)
txn_counts.plot(kind="bar", rot=0)
plt.title("Transaction Types")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()

cleaned_data = data.copy()
print(cleaned_data.head())

text_cols = cleaned_data.select_dtypes(include="object").columns
print(text_cols)

# Encode text data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in text_cols:
    cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))

print(cleaned_data.info())
print(cleaned_data.head())

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(df):
    vif_df = pd.DataFrame()
    vif_df["vars"] = df.columns
    vif_df["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_df

print(calc_vif(cleaned_data))

# Create new features
cleaned_data['amt_change_orig'] = cleaned_data.apply(lambda x: x['oldbalanceOrg'] - x['newbalanceOrig'], axis=1)
cleaned_data['amt_change_dest'] = cleaned_data.apply(lambda x: x['oldbalanceDest'] - x['newbalanceDest'], axis=1)
cleaned_data['txn_path'] = cleaned_data.apply(lambda x: x['nameOrig'] + x['nameDest'], axis=1)

# Drop unnecessary columns
cleaned_data = cleaned_data.drop(['oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','step','nameOrig','nameDest'], axis=1)

print(calc_vif(cleaned_data))

corr_mat = cleaned_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_mat, annot=True)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Scale amount
scaler = StandardScaler()
cleaned_data["norm_amount"] = scaler.fit_transform(cleaned_data["amount"].values.reshape(-1, 1))
cleaned_data.drop(["amount"], inplace=True, axis=1)

target = cleaned_data["isFraud"]
features = cleaned_data.drop(["isFraud"], axis=1)

# Split data
(X_train, X_test, y_train, y_test) = train_test_split(features, target, test_size=0.3, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Train decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

dt_preds = dt.predict(X_test)
dt_score = dt.score(X_test, y_test) * 100

# Train random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)
rf_score = rf.score(X_test, y_test) * 100

print("DT Score:", dt_score)
print("RF Score:", rf_score)

# Confusion matrix - DT
print("DT Confusion Matrix")
tn, fp, fn, tp = confusion_matrix(y_test, dt_preds).ravel()
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

# Confusion matrix - RF
print("RF Confusion Matrix")
tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

dt_cm = confusion_matrix(y_test, dt_preds.round())
print("DT Confusion Matrix")
print(dt_cm)

rf_cm = confusion_matrix(y_test, rf_preds.round())
print("RF Confusion Matrix")
print(rf_cm)

dt_report = classification_report(y_test, dt_preds)
print("DT Classification Report")
print(dt_report)

rf_report = classification_report(y_test, rf_preds)
print("RF Classification Report")
print(rf_report)

# Plot DT confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm)
disp.plot()
plt.title('DT Confusion Matrix')
plt.show()

# Plot RF confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm)
disp.plot()
plt.title('RF Confusion Matrix')
plt.show()

# DT ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, dt_preds)
auc = metrics.auc(fpr, tpr)

plt.title('DT ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# RF ROC curve
fpr, tpr, _ = metrics.roc_curve(y_test, rf_preds)
auc = metrics.auc(fpr, tpr)

plt.title('RF ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()