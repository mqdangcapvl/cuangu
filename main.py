import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/kickstarter_projects.csv')
print(df.head())
print(df.describe())
print(df.columns)

df = df.dropna()

df.columns = df.columns.str.lower()
df['launched'] = pd.to_datetime(df['launched'])
df['deadline'] = pd.to_datetime(df['deadline'])

df['duration'] = (df['deadline'] - df['launched']).dt.days
df = df[df['goal'] > 0 ]

df['state'] = df['state'].str.lower()
df['success'] = df['state'].apply(lambda x: 1 if x == 'successful' else 0)
success_rate = df.groupby('category')['success'].mean().sort_values(ascending=False)
print(success_rate.head())
top10 = success_rate.head(10)

num_success = df[df['success'] == 1].shape[0]
print("Number of successful projects:", num_success)

df['completion'] = df['pledged'] / df['goal']
filtered = df[df['goal'] > 1000]
top_project = filtered.sort_values(by='completion', ascending=False).iloc[0]

print("Top project (goal > 1000):")
print(top_project[['name', 'goal', 'pledged', 'completion']])

df['year'] = df['launched'].dt.year
yearly = df.groupby(['year', 'state']).size().unstack()
yearly['success_more_than_fail'] = yearly['successful'] > yearly['failed']
print(yearly)

country = df.groupby(['country', 'state'])['pledged'].sum().unstack()
country['fail_more_than_success'] = country['failed'] > country['successful']
print(country)

plt.figure()
top10.plot(kind='bar')
plt.title("Top 10 Category Success Rate")
plt.xlabel("Category")
plt.ylabel("Success Rate")
plt.xticks(rotation=45)
plt.tight_layout()

yearly[['successful', 'failed']].plot(kind='line')
plt.title("Successful vs Failed Projects Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Projects")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = df[['goal', 'duration', 'backers']]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))