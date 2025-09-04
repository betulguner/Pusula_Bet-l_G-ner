import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.pipeline import Pipeline



DATA_PATH = "/Users/betulguner/Documents/dataCaseStudy/Talent_Academy_Case_DT_2025.xlsx"
SAVE_PATH = "/Users/betulguner/Documents/dataCaseStudy/cleaned_data.csv"
PLOTS_DIR = "/Users/betulguner/Documents/dataCaseStudy/plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

#Excel dosyasını oku
def load_data(path):
    return pd.read_excel(path)


def process_target(df):
    """TedaviSuresi'ni numerik hale getir"""
    df["TedaviSuresi_num"] = df["TedaviSuresi"].str.extract(r"(\d+)").astype(float)
    return df

def plot_target_distribution(df, save_dir):
    """Target dağılım görselleri"""
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    df["TedaviSuresi_num"].hist(bins=20, color="skyblue", edgecolor="black")
    plt.title("Tedavi Süresi Histogram")
    plt.subplot(1,2,2)
    sns.kdeplot(df["TedaviSuresi_num"].dropna(), fill=True, color="salmon")
    plt.title("Tedavi Süresi Distribution")
    plt.savefig(os.path.join(save_dir, "tedavi_suresi_plot.png"))
    plt.close()

def remove_duplicates(df):
    """Duplicate satırları kaldır"""
    duplicate_rows = df[df.duplicated()]
    print("Toplam duplicate satır:", duplicate_rows.shape[0])
    df_no_dupes = df.drop_duplicates()
    print("Drop sonrası shape:", df_no_dupes.shape)
    return df_no_dupes


def analyze_missing(df, save_dir):
    """Eksik değer analizi ve heatmap"""
    missing_values = df.isnull().sum().sort_values(ascending=False)
    print("\nEksik değer sayıları:\n", missing_values)
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Eksik Değer Heatmap")
    plt.savefig(os.path.join(save_dir, "missing_values_heatmap.png"))
    plt.close()

from sklearn.impute import SimpleImputer, KNNImputer

def fill_missing(df):
    """Eksik değer doldurma stratejisi (gelişmiş)"""
    # 1. Uygulama süresi numerik hale getir
    df["UygulamaSuresi_num"] = df["UygulamaSuresi"].str.extract(r"(\d+)").astype(float)

    # Kolon grupları
    categoricals = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum", "TedaviAdi"]
    multi_cols = ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri"]
    numerics = ["Yas", "UygulamaSuresi_num"]

    # 2. Kategorik kolonları mode (most_frequent) ile doldur
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categoricals] = cat_imputer.fit_transform(df[categoricals])

    # 3. Çoklu kolonları özel olarak doldur
    # Eğer NaN ise "Bilinmiyor" yaz
    for col in multi_cols:
        df[col] = df[col].fillna("Bilinmiyor")

    # 4. Sayısal kolonları KNNImputer ile doldur
    knn_imputer = KNNImputer(n_neighbors=5)
    df[numerics] = knn_imputer.fit_transform(df[numerics])

    print("\nEksik değer doldurma sonrası tekrar kontrol:")
    print(df.isnull().sum())
    return df


def analyze_features(df):
    """Feature tiplerini analiz et"""
    categoricals = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum", "TedaviAdi"]
    multi_cols = ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri"]
    numerics = ["Yas", "UygulamaSuresi_num"]

    for col in categoricals:
        print(f"\n{col} unique values:", df[col].nunique())

    for col in multi_cols:
        all_tokens = df[col].str.split(",").sum()
        token_counts = Counter(all_tokens)
        print(f"\n{col} unique tokens ({len(token_counts)}): {list(token_counts.keys())[:10]} ...")

    print("\nSayısal değişken özetleri:")
    print(df[numerics].describe())
    return categoricals, multi_cols, numerics

def plot_eda(df, categoricals, numerics, save_dir):
    """EDA görselleri"""
    for col in numerics:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, color="skyblue")
        plt.title(f"{col} Histogram & Distribution")
        plt.savefig(os.path.join(save_dir, f"{col}_hist.png"))
        plt.close()

    for col in categoricals:
        plt.figure(figsize=(6,4))
        sns.countplot(y=df[col],
                      order=df[col].value_counts().index,
                      palette="viridis")
        plt.title(f"{col} Distribution")
        plt.savefig(os.path.join(save_dir, f"{col}_bar.png"))
        plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(df[numerics].corr(), annot=True, cmap="coolwarm")
    plt.title("Sayısal Değişkenler Korelasyon Heatmap")
    plt.savefig(os.path.join(save_dir, "numerics_corr_heatmap.png"))
    plt.close()

    print("\nEDA görselleri plots/ klasörüne kaydedildi ✅")
    

def preprocess_data(df):
    # Drop kullanılmayan kolonlar
    df = df.drop(columns=["HastaNo", "TedaviSuresi", "UygulamaSuresi"])
    
    # Multi-label kolonları listelere çevir
    for col in ["KronikHastalik", "Alerji", "Tanilar"]:
        df[col] = df[col].apply(lambda x: [i.strip() for i in str(x).split(",")])
    
    # MultiLabelBinarizer uygula
    mlb = {}
    mlb_features = []
    for col in ["KronikHastalik", "Alerji", "Tanilar"]:
        mlb[col] = MultiLabelBinarizer()
        transformed = mlb[col].fit_transform(df[col])
        mlb_df = pd.DataFrame(transformed, columns=[f"{col}_{cls}" for cls in mlb[col].classes_])
        mlb_features.append(mlb_df)
    df = df.drop(columns=["KronikHastalik", "Alerji", "Tanilar"])
    df = pd.concat([df.reset_index(drop=True)] + mlb_features, axis=1)
    
    # Kategorik kolonlar
    categoricals = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum", "TedaviAdi", "UygulamaYerleri"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_df = pd.DataFrame(ohe.fit_transform(df[categoricals]),
                          columns=ohe.get_feature_names_out(categoricals))
    
    df = df.drop(columns=categoricals)
    df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
    
    # Numerik kolonları scale et
    scaler = StandardScaler()
    numerics = ["Yas", "UygulamaSuresi_num", "TedaviSuresi_num"]
    df[numerics] = scaler.fit_transform(df[numerics])
    
    return df

  
def save_cleaned_data(df, path):
    """Temiz dataset'i kaydet"""
    df.to_csv(path, index=False)
    print(f"\nCleaned dataset kaydedildi: {path}")
    
    
    
def run_pipeline():
    df = load_data(DATA_PATH)
    df = process_target(df)
    plot_target_distribution(df, PLOTS_DIR)

    df = remove_duplicates(df)
    analyze_missing(df, PLOTS_DIR)
    df = fill_missing(df)

    categoricals, multi_cols, numerics = analyze_features(df)
    plot_eda(df, categoricals, numerics, PLOTS_DIR)

    df = preprocess_data(df)
    
    save_cleaned_data(df, SAVE_PATH)
    
    X = df.drop(columns=["TedaviSuresi_num"])  
    y = df["TedaviSuresi_num"] 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42) 

    print("\nTrain-Test split tamamlandı ✅")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    

if __name__ == "__main__":
    run_pipeline()
    
    


