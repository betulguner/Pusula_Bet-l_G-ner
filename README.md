
# Pusula_Betul_Guner
# Data Science Intern Case Study  
**Name:** Betül Güner  
**Email:** gunerbetul14@gmail.com  

Bu çalışma, Fiziksel Tıp ve Rehabilitasyon veriseti üzerinde **EDA** ve **veri ön işleme** adımlarını kapsar.  
Hedef değişken: `TedaviSuresi`.

---

## 📌 İçerik
- [Veri Seti Hakkında](#veri-seti-hakkında)
- [Kolon Açıklamaları](#kolon-açıklamaları)
- [Pipeline](#pipeline)
- [EDA](#eda)
- [Eksik Değer İşlemleri](#eksik-değer-işlemleri)
- [Özellik Mühendisliği](#özellik-mühendisliği)
- [Train-Test Split](#train-test-split)
- [Klasör Yapısı](#klasör-yapısı)
- [How to Run](#how-to-run)

---

## Veri Seti Hakkında
- **Toplam gözlem**: 2235  
- **Kalan gözlem (temizleme sonrası)**: 1307  
- **Özellik sayısı**: 13 (hedef dahil)  
- **Hedef değişken**: `TedaviSuresi`

---

## Kolon Açıklamaları
- **HastaNo**: Anonymized patient ID  
- **Yas**: Age  
- **Cinsiyet**: Gender  
- **KanGrubu**: Blood type  
- **Uyruk**: Nationality  
- **KronikHastalik**: Chronic conditions (comma-separated list)  
- **Bolum**: Department/Clinic  
- **Alerji**: Allergies (may be single or comma-separated)  
- **Tanilar**: Diagnoses  
- **TedaviAdi**: Treatment name  
- **TedaviSuresi**: Treatment duration in sessions  
- **UygulamaYerleri**: Application sites  
- **UygulamaSuresi**: Application duration  

---

## Pipeline
Tüm adımlar `preprocessing.py` içinde fonksiyonel yapıda uygulanmıştır:

1. **Target Dönüştürme** → `TedaviSuresi` → `TedaviSuresi_num` (örn. "15 Seans" → 15)  
2. **Duplicate Analizi** → 928 tam duplicate silindi.  
3. **Eksik Değer Analizi & Doldurma**  
   - Kategorikler → Mod (en sık görülen)  
   - Multi-label kolonlar → `"Bilinmiyor"`  
   - Sayısal kolonlar → `KNNImputer`  
4. **Özellik Mühendisliği**  
   - Multi-label kolonlar (`Alerji`, `KronikHastalik`, `Tanilar`) → `MultiLabelBinarizer`  
   - Kategorik kolonlar → `OneHotEncoder`  
   - Numerikler → `StandardScaler`  
5. **EDA Görselleştirme** → `plots/` klasöründe saklandı.  
6. **Train-Test Split**  
   - `X_train`: (1045, 637)  
   - `X_test`: (262, 637)  
   - `y_train`: (1045,)  
   - `y_test`: (262,)  

---

## EDA
- **Eksik Değer Analizi**: heatmap ve bar grafikleri  
- **Sayısal Değişkenler**: histogram + KDE  
- **Kategorik Değişkenler**: countplot  
- **Korelasyon Analizi**: heatmap  

Tüm görseller → `plots/` klasöründe.

---

## Eksik Değer İşlemleri
Örnek:  
- `KanGrubu` → mod ile dolduruldu.  
- `KronikHastalik`, `Alerji`, `Tanilar` → `"Bilinmiyor"` etiketi eklendi.  
- `UygulamaSuresi_num` → KNN imputation.  

---

## Özellik Mühendisliği
- Çok etiketli kolonlar genişletildi (`MultiLabelBinarizer`)  
- Kategorikler OHE ile encode edildi.  
- Numerikler `StandardScaler` ile normalize edildi.  

---

## Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

---

## How to Run
```bash
git clone https://github.com/username/Pusula_Betul_Guner.git
cd dataCaseStudy
python -m preprocessing
<<<<<<< HEAD

=======
>>>>>>> a200d33 (EDA + preprocessing pipeline + README + plots)
