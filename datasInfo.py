import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# -------------  AYAR: dosya yolunu kontrol et -------------
path = "/Users/betulguner/Documents/dataCaseStudy/Talent_Academy_Case_DT_2025.xlsx"
# Eğer dosyada birden fazla sheet varsa listeler:
xls = pd.ExcelFile(path)
print("Available sheets:", xls.sheet_names)

# varsayılan ilk sheet'i oku (gerekirse sheet_name='Sheet1' olarak değiştir)
df = pd.read_excel(path, sheet_name=0, engine="openpyxl")

# -------------  TEMEL KONTROLLER -------------
print("\n--- SHAPE ---")
print("df.shape:", df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string(index=False))

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIBE (all) ---")
print(df.describe(include='all').T)

print("\n--- NULLS (top 50) ---")
nulls = df.isnull().sum().sort_values(ascending=False)
print(nulls.head(50))

print("\n--- DUPLICATES ---")
print("Duplicate rows count:", df.duplicated().sum())

# -------------  KOLONLAR VE UNIQUE SAYILARI -------------
print("\n--- COLUMNS & UNIQUE COUNTS ---")
for c in df.columns:
    try:
        print(f"{c:20} : {df[c].nunique()} unique")
    except Exception as e:
        print(c, "error:", e)

# -------------  TARGET (TedaviSuresi) ANALIZI -------------
if 'TedaviSuresi' in df.columns:
    print("\n--- TedaviSuresi value_counts (top 30) ---")
    print(df['TedaviSuresi'].value_counts(dropna=False).head(30))
    print("\nTedaviSuresi describe:")
    print(df['TedaviSuresi'].describe())
    # Basit histogram
    try:
        plt.figure(figsize=(6,3))
        df['TedaviSuresi'].dropna().astype(float).hist(bins=30)
        plt.title('TedaviSuresi histogram')
        plt.show()
    except Exception as e:
        print("Histogram çizilemedi:", e)
else:
    print("\n!!! Dataset'te 'TedaviSuresi' sütunu bulunamadı. Sütun adını kontrol et. !!!")

# -------------  POTANSİYEL HATA KAYNAKLARINI KONTROL ET -------------
print("\n--- Debug checks that help find the 'small number of rows' problem ---")

X = df.drop(columns=['TedaviSuresi']) if 'TedaviSuresi' in df.columns else df.copy()
y = df['TedaviSuresi'] if 'TedaviSuresi' in df.columns else pd.Series(dtype='float')
print("X.shape before any encoding:", X.shape)
print("y.shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("After sklearn train_test_split:")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)


print("\nSample index values (first 20):", list(df.index[:20]))
print("First 10 rows of df.shape and sample counts per column:")
print(df.head(10).T)
