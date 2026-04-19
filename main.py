# import thu vien
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# doc file
df = pd.read_csv('data/kickstarter_projects.csv')
print(df.head())
print(df.describe())

# lam sach dlieu
df = df.drop_duplicates() # xoa cac dong trung lap

df['Launched'] = pd.to_datetime(df['Launched'], errors='coerce') # chuyen doi cot Launched sang dang datetime
df['Deadline'] = pd.to_datetime(df['Deadline'], errors='coerce') # chuyen doi cot Deadline sang dang datetime

df = df.dropna(subset=['Launched', 'Deadline']) # xoa cac dong co gia tri ngay thang khong hop le
df = df.dropna(subset=['Goal', 'Pledged', 'Backers', 'State', 'Category', 'Country']) # xoa cac dong co gia tri khong hop le trong cac cot Goal, Pledged, Backers, State, Category, Country

df['duration'] = (df['Deadline'] - df['Launched']).dt.days # tinh toan thoi gian ke hoach du an va luu vao cot moi 'duration'
df = df[df['duration'] > 0] # loc cac dong co gia tri thoi gian khong hop le (duration <= 0)

df = df[(df['Goal'] > 0) & (df['Pledged'] >= 0) & (df['Backers'] >= 0)]  # loc cac dong co gia tri khong hop le trong cot Goal, Pledged, Backers (Goal phai lon hon 0, Pledged va Backers phai lon hon hoac bang 0)
df['success'] = (df['State'] == 'Successful').astype(int) # tao cot moi 'success' de bieu dien tinh trang thanh cong cua du an (1 neu du an thanh cong, 0 neu khong thanh cong)

success_rate = df.groupby('Category')['success'].mean().sort_values(ascending=False)
top10 = success_rate.head(10)
print("Top categories by success rate:")
print(top10)

num_success = df['success'].sum()
print("Number of successful projects:", num_success)

df['completion'] = df['Pledged'] / df['Goal']
filtered = df[df['Goal'] > 1000]
top_project = filtered.sort_values(by='completion', ascending=False).iloc[0]
print("\nTop project (Goal > 1000):")
print(top_project[['Name', 'Goal', 'Pledged', 'completion']])

df['year'] = df['Launched'].dt.year
yearly = df.groupby(['year', 'State']).size().unstack()
yearly = yearly.fillna(0)
yearly['success_more_than_fail'] = yearly['Successful'] > yearly['Failed']
print("\nYearly comparison:")
print(yearly)

country = df.groupby(['Country', 'State'])['Pledged'].sum().unstack()
country = country.fillna(0)
country['fail_more_than_success'] = country['Failed'] > country['Successful']
print("\nCountry comparison:")
print(country)

plt.figure(figsize=(10, 6))
top10.plot(kind='bar')
plt.title("Top 10 Category Success Rate")
plt.xlabel("Category")
plt.ylabel("Success Rate")
plt.xticks(rotation=45)
plt.tight_layout()

yearly[['Successful', 'Failed']].plot()
plt.title("Successful vs Failed Projects Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Projects")

plt.show()