# 1. import thu vien va doc du lieu
import numpy as np  # thu vien tinh toan
import pandas as pd # thu vien xu ly du lieu
import matplotlib.pyplot as plt # thu vien ve do thi

df = pd.read_csv('data/kickstarter_projects.csv') # doc du lieu tu file csv vao df
print(df.head())  # hien thi 5 dong dau tien cua df
print(df.describe()) # hien thi thong ke co ban cua cac cot so trong df
# ===============================

# 2. xu ly du lieu
df = df.dropna()  # loai bo cac dong co gia tri bi thieu

df['Launched'] = pd.to_datetime(df['Launched']) # chuyen cot 'Launched' sang kieu datetime
df['Deadline'] = pd.to_datetime(df['Deadline']) # tuong tu cot launched

df['duration'] = (df['Deadline'] - df['Launched']).dt.days # tinh so ngay giua deadline va launched de tao cot duration
df = df[df['Goal'] > 0 ]   # loai bo cac dong co gia tri goal <= 0
# ===============================

# 3. phan tich du lieu va ti le thanh cong theo category
df['success'] = df['State'].apply(lambda x: 1 if x == 'Successful' else 0) # tao bien binary ten la success: 1 neu state la successful, nguoc lai la 0
success_rate = df.groupby('Category')['success'].mean().sort_values(ascending=False) # tinh ty le thanh cong theo category va sap xep giam dan
top10 = success_rate.head(10) # lay 10 category co ty le thanh cong cao nhat
print(success_rate.head()) # hien thi 5 category co ty le thanh cong cao nhat
# ===============================

# 4. dem so du an thanh cong va that bai, tim du an co ty le thanh cong cao nhat, va so sanh so du an thanh cong va that bai theo nam va theo quoc gia
num_success = df[df['success'] == 1].shape[0]  # dem so du an thanh cong bang cach loc df theo cot success = 1 va dem so dong
print("Number of successful projects:", num_success) # hien thi so du an thanh cong
# ===============================

# 5. tim du an co ty le thanh cong cao nhat trong so nhung du an co goal > 1000
df['completion'] = df['Pledged'] / df['Goal'] # tinh ty le hoan thanh bang cach chia cot Pledged cho cot Goal
filtered = df[df['Goal'] > 1000]  # loc df de chi con nhung du an co goal > 1000
top_project = filtered.sort_values(by='completion', ascending=False).iloc[0] # sap xep df theo cot completion giam dan va lay dong dau tien

print("Top project (Goal > 1000):") # in
print(top_project[['Name', 'Goal', 'Pledged', 'completion']]) # hien thi ten, goal, pledged, va ty le hoan thanh cua du an co ty le hoan thanh cao nhat trong so nhung du an co goal > 1000
# ================================

# 6. so sanh so du an thanh cong va that bai theo nam va theo quoc gia 
df['year'] = df['Launched'].dt.year # tao cot year tu cot Launched de phan tich theo nam
yearly = df.groupby(['year', 'State']).size().unstack()  # dem so du an theo nam va state, sau do unstack de chuyen state thanh cot
yearly['success_more_than_fail'] = yearly['Successful'] > yearly['Failed'] # tao cot s_m_t_f de so sanh xem so du an thanh cong co nhieu hon so du an that bai hay khong theo nam
print(yearly)
# =================================

# 7. so sanh so du an thanh cong va that bai theo quoc gia
country = df.groupby(['Country', 'State'])['Pledged'].sum().unstack() # dem tong so tien pledged theo quoc gia va state r unstack
country['fail_more_than_success'] = country['Failed'] > country['Successful'] # tao cot f_m_t_s de so sanh xem tong so tien pledged cua nhung du an that bai co nhieu hon tong so tien pledged cua nhung du an thanh cong hay ko theo quoc gia
print(country) # in ket qua so sanh theo quoc gia
# =================================

# 8. ve do thi
plt.figure(figsize=(10, 6)) # tao figure co kich thuoc 10x6 inch
top10.plot(kind='bar') # ve bar chart
plt.title("Top 10 Category Success Rate") # title 
plt.xlabel("Category") # ten truc x 
plt.ylabel("Success Rate") # ten truc y
plt.xticks(rotation=45) # xoay ten 45 do de de doc hon
plt.tight_layout() # dieu chinh layout de tranh bi cat

yearly[['Successful', 'Failed']].plot(kind='line') # ve do thi duong cho so du an thanh cong va that bai theo nam
plt.title("Successful vs Failed Projects Over Years") # title
plt.xlabel("Year") # ten truc x
plt.ylabel("Number of Projects") # ten truc y
plt.show() # in do thi
# ====================================

# 9. xay dung mo hinh du doan su thanh cong cua du an dua tren goal, duration, va so luong backers
from sklearn.model_selection import train_test_split # import ham chia du lieu thanh tap train va test
from sklearn.linear_model import LogisticRegression  # import mo hinh logistic regression
from sklearn.metrics import accuracy_score # import ham tinh do chinh xac cua mo hinh

X = df[['Goal', 'duration', 'Backers']] # chon cac cot Goal, duration, backers lam bien dau vao X
y = df['success'] # bien muc tieu la cot success trong df, luu vao y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # chia du lieu thanh tap train va test, trong do 20% du lieu duoc su dung lam tap test

model = LogisticRegression() # khoi tao mo hinh logistic regression
model.fit(X_train, y_train) # huan luyen mo hinh tren tap train, trong do X_train la bien dau vao va y_train la bien muc tieu

y_pred = model.predict(X_test) # du doan ket qua tren tap test, luu ket qua vao y_pred

print("Accuracy:", accuracy_score(y_test, y_pred)) # tinh do chinh xac cua mo hinh bang cach so sanh ket qua du doan y_pred voi gia tri thuc te y_test, va in ket qua