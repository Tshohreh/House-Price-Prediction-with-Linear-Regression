import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ایجاد دیتاست
data = {
    'Area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],  # مساحت خانه
    'Rooms': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],              # تعداد اتاق‌ها
    'Price': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550]  # قیمت خانه
}
df = pd.DataFrame(data)

# نمایش دیتاست
print(df)


# انتخاب ویژگی‌ها (X) و هدف (y)
X = df[['Area', 'Rooms']]  # ویژگی‌ها
y = df['Price']            # هدف

# تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# ایجاد مدل رگرسیون خطی
model = LinearRegression()

# آموزش مدل
model.fit(X_train, y_train)

# نمایش ضرایب مدل
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# پیش‌بینی روی داده‌های آزمون
y_pred = model.predict(X_test)

# محاسبه خطا
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# مقایسه خروجی واقعی و پیش‌بینی‌شده
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
