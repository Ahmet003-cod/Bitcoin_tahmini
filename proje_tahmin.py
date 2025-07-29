import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
import numpy as np

df=pd.read_excel("C:/Users\Huzur Bilgisayar/OneDrive\Masaüstü/önemli kodlar/No NLP-Dataset.xlsx")

df.head()
df.drop(["Date","Year","Month","Day","Day_of_Week", "Is_Weekend", "Quarter","Week_of_Year", "Is_Holiday"],axis=1,inplace=True)

X=df.drop("Bitcoin",axis=1)
y=df["Bitcoin"]

scaler=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
X_train_scaler=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.2, max_depth=4)
model.fit(X_train_scaler, y_train)
y_pred=model.predict(X_test_scaler)
print("xgboost MSE:", mean_squared_error(y_test, y_pred))
print("xgboost R2 Score:", r2_score(y_test, y_pred))

"""X_today = np.array([[67.35, 99.08, 3313.3, 44850, 6405, 21286, 4.40, 14.80, 118615, 119000, 118000]])
X_scaled = scaler.transform(X_today)
y_hat = model.predict(X_scaled)
print("Tahmini Bitcoin Fiyatı (29 Tem. 2025):", y_hat[0])



from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=200,random_state=42)
rfr.fit(X_train_scaler,y_train)
y_pred=rfr.predict(X_test_scaler)

print("RandomForestRegressor mean_squared_error= ",mean_squared_error(y_test,y_pred))
print("RandomForestRegressor r2_score= ",r2_score(y_test,y_pred))



X_today = np.array([[67.35, 99.08, 3313.3, 44850, 6405, 21286, 4.40, 14.80, 118615, 119000, 118000]])
X_scaled = scaler.transform(X_today)
y_hat = rfr.predict(X_scaled)
print("Tahmini Bitcoin Fiyatı (29 Tem. 2025):", y_hat[0])


import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train_scaler, y_train)
y_pred = model.predict(X_test_scaler)

print("LightGBM MSE:", mean_squared_error(y_test, y_pred))
print("LightGBM R2 Score:", r2_score(y_test, y_pred))


"""






import tkinter as tk
from tkinter import messagebox
import numpy as np

# Daha önce eğittiğin model ve scaler objeleri
# lr, rfr, scaler gibi nesneleri burada kullanacağız.
# Bu örnekte rfr ve scaler'ı global değişken olarak varsayıyoruz.

def predict_bitcoin():
    try:
        # Kullanıcıdan girişleri al, float'a çevir
        inputs = []
        for entry in entries:
            val = float(entry.get())
            inputs.append(val)
        
        # Numpy array'e dönüştür, ölçeklendir
        X_new = np.array(inputs).reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        
        # Tahmin yap
        y_pred = model.predict(X_new_scaled)
        
        # Sonucu göster
        result_var.set(f"Tahmini Bitcoin Fiyatı: {y_pred[0]:.2f}")
        
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm kutucuklara geçerli sayılar girin!")

# Tkinter pencere oluştur
root = tk.Tk()
root.title("Bitcoin Fiyat Tahmini")

# Özellik isimleri
features = ['Oil', 'DXY', 'Gold', 'DJI', 'SP500', 'IXIC', 'TNX', 'VIX',
            'Bitcoin_Lag1', 'Bitcoin_Lag2', 'Bitcoin_Lag3']

entries = []

# Form oluştur
for i, feature in enumerate(features):
    tk.Label(root, text=feature).grid(row=i, column=0, padx=5, pady=5, sticky='w')
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

# Tahmin butonu
predict_button = tk.Button(root, text="Tahmin Et", command=predict_bitcoin)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=10)

# Sonuç gösterimi
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 14))
result_label.grid(row=len(features)+1, column=0, columnspan=2, pady=10)

root.mainloop()
