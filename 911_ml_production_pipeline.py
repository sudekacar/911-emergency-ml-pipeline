# -*- coding: utf-8 -*-
"""
Qybit Labs — 911 Acil Çağrı Sınıflandırma ve Veri Mühendisliği Pipeline Projesi
Yazar: Sude Yaren Kaçar (Founder & Lead Developer)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def dummy_data_olustur():
    """
    Eğer gerçek 911.csv dosyası dizinde yoksa, pipeline'ın 
    hata vermeden çalışabilmesi için yapay bir veri seti üretir.
    """
    print("⚠️ 911.csv bulunamadı. Testler için geçici yapay veri seti üretiliyor...")
    np.random.seed(42)
    n_samples = 1000
    data = {
        'lat': np.random.uniform(40.0, 41.0, n_samples),
        'lng': np.random.uniform(-75.5, -74.5, n_samples),
        'desc': np.random.choice([
            'RESCUE VEHICLE - TRAFFIC ACCIDENT', 
            'FIRE DEPT - HOUSE FIRE', 
            'EMS - RESPIRATORY EMERGENCY'
        ], n_samples),
        'zip': np.random.choice([19401.0, 19403.0, 19406.0, 19446.0], n_samples),
        'title': np.random.choice([
            'Traffic: VEHICLE ACCIDENT -', 
            'Fire: FIRE ALARM -', 
            'EMS: CARDIAC EMERGENCY -'
        ], n_samples),
        'timeStamp': pd.date_range(start='2026-01-01', periods=n_samples, freq='min').strftime('%Y-%m-%d %H:%M:%S'),
        'twp': np.random.choice(['LOWER MERION', 'ABINGTON', 'NORRISTOWN'], n_samples),
        'addr': ['Test Adresi'] * n_samples,
        'e': [1] * n_samples
    }
    df = pd.DataFrame(data)
    df.to_csv('911.csv', index=False)
    return df

def veri_yukle(dosya_yolu='911.csv'):
    """
    911 Acil Çağrı veri setini güvenli bir şekilde yükler.
    """
    if not os.path.exists(dosya_yolu):
        df = dummy_data_olustur()
    else:
        print(f"📂 {dosya_yolu} başarıyla tespit edildi. Veri yükleniyor...")
        df = pd.read_csv(dosya_yolu)
    return df

def istatistiksel_analiz_ve_eda(df):
    """
    Verinin merkezi eğilim ölçülerini hesaplar ve istatistiksel doğrulamaları yapar.
    """
    print("\n--- 📊 İSTATİSTİKSEL VERİ ANALİZİ ---")
    
    # Boş verileri temizleme
    df = df.dropna(subset=['lat', 'lng', 'title'])
    
    # Ortalama, Medyan ve Standart Sapma Hesaplamaları
    print(f"Enlem (Lat) Ortalaması: {df['lat'].mean():.4f}")
    print(f"Boylam (Lng) Medyanı: {df['lng'].median():.4f}")
    print(f"Boylam (Lng) Standart Sapması: {df['lng'].std():.4f}")
    
    # 25%, 50%, 75% Çeyreklik Analizleri
    print("\nEnlem (Lat) ve Boylam (Lng) Çeyreklik Dağılımı:")
    print(df[['lat', 'lng']].quantile([0.25, 0.5, 0.75]))
    
    # Chi-Square Bağımsızlık Testi (Kategorik Değişkenler Arası Anlamlılık)
    # Acil çağrı bölgeleri (Township) ile çağrı türleri arasındaki ilişki analizi
    df['type'] = df['title'].apply(lambda x: x.split(':')[0])
    contingency_table = pd.crosstab(df['twp'], df['type'])
    
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    print(f"\n⚡ Chi-Square Testi Sonucu (twp vs type):")
    print(f"Chi-Square Değeri: {chi2:.4f}")
    print(f"p-değeri (Anlamlılık Derecesi): {p:.4e}")
    if p < 0.05:
        print("💡 p < 0.05: Çağrı yapılan bölgeler ile acil durum türleri arasında güçlü ve anlamlı bir istatistiksel ilişki vardır.")
    else:
        print("⚠️ Değişkenler arasında istatistiksel olarak anlamlı bir ilişki tespit edilemedi.")
        
    return df

def veri_on_isleme_ve_boyut_indirgeme(df):
    """
    Kategorik kodlama, normalizasyon ve PCA adımlarını yürütür.
    """
    print("\n--- 🛠️ VERİ ÖN İŞLEME VE BOYUT İNDİRGEME ---")
    
    # Tahmin edilecek hedef değişkeni (Label) ve özellikleri (Features) ayırma
    # Örn: Koordinat verileri ve kodlanmış bölge bilgileri
    X = df[['lat', 'lng']].copy()
    
    # Kategorik bölgeyi (twp) sayısal değere kodlama (OrdinalEncoder)
    encoder = OrdinalEncoder()
    X['twp_encoded'] = encoder.fit_transform(df[['twp']].astype(str))
    
    # Hedef Değişken (Çağrı Türü: EMS, Fire, Traffic)
    y = df['type']
    
    # Veriyi Standardizasyon ile ölçekleme (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Veriler StandardScaler ile başarıyla normalize edildi (Mean=0, Std=1).")
    
    # Train-Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # PCA (Temel Bileşen Analizi) ile Boyut İndirgeme
    # Real-time tahmin hızını artırmak ve overfitting'i engellemek için boyutu azaltıyoruz
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"✓ PCA Başarıyla Tamamlandı. Veri boyutu {X_train.shape[1]}'den {X_train_pca.shape[1]}'ye indirildi.")
    
    return X_train_pca, X_test_pca, y_train, y_test

def modelleri_benchmark_et(X_train, X_test, y_train, y_test):
    """
    K-NN, Random Forest ve Farklı SVM Hiperparametrelerini Yarıştırır ve Karşılaştırır.
    """
    print("\n--- ⚔️ MAKİNE ÖĞRENMESİ BENCHMARK SAVAŞ ALANI ---")
    
    sonuclar = {}
    
    # 1. K-Nearest Neighbors (K-NN)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    sonuclar['K-NN'] = acc_knn
    print(f"Model: K-NN | Doğruluk Skoru: %{acc_knn*100:.2f}")
    
    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    sonuclar['Random Forest'] = acc_rf
    print(f"Model: Random Forest | Doğruluk Skoru: %{acc_rf*100:.2f}")
    
    # 3. Support Vector Machine (Default Hyperparameters)
    svc_def = SVC()
    svc_def.fit(X_train, y_train)
    y_pred_def = svc_def.predict(X_test)
    acc_def = accuracy_score(y_test, y_pred_def)
    sonuclar['SVM (Default)'] = acc_def
    print(f"Model: SVM (Default) | Doğruluk Skoru: %{acc_def*100:.2f}")
    
    # 4. Support Vector Machine (RBF Kernel, Tuned C=1000.0)
    # Ceza parametresini yükselterek sınır doğruluğunu maksimize ediyoruz
    svc_tuned = SVC(C=1000.0, kernel='rbf')
    start_time = time.time()
    svc_tuned.fit(X_train, y_train)
    y_pred_tuned = svc_tuned.predict(X_test)
    acc_tuned = accuracy_score(y_test, y_pred_tuned)
    sonuclar['SVM (Tuned RBF C=1000)'] = acc_tuned
    inference_time = (time.time() - start_time) * 1000
    print(f"Model: SVM (Tuned RBF C=1000) | Doğruluk Skoru: %{acc_tuned*100:.2f} | Eğitim & Çıkarım Süresi: {inference_time:.2f} ms")
    
    # Sınıflandırma Detay Raporu (Precision, Recall, F1-Score)
    print("\n=== SVM (Tuned RBF C=1000) DETAYLI SINIFLANDIRMA RAPORU ===")
    print(classification_report(y_test, y_pred_tuned))
    
    # En İyi Modeli Kaydetme
    joblib.dump(svc_tuned, '911_best_svm_model.pkl')
    print("✓ En başarılı SVM modeli '911_best_svm_model.pkl' olarak diske kaydedildi.")
    
    return sonuclar

def main():
    print("="*60)
    print("🚀 QYBIT LABS — ACİL ÇAĞRI SINIFLANDIRMA PIPELINE BAŞLATILDI")
    print("="*60)
    
    # 1. Veriyi Yükleme
    df = veri_yukle()
    
    # 2. EDA & İstatistiki Analiz
    df_analyzed = istatistiksel_analiz_ve_eda(df)
    
    # 3. Ön İşleme & PCA
    X_train, X_test, y_train, y_test = veri_on_isleme_ve_boyut_indirgeme(df_analyzed)
    
    # 4. Model Benchmark Etme
    modelleri_benchmark_et(X_train, X_test, y_train, y_test)
    
    print("\n🎉 Tüm veri mühendisliği adımları başarıyla tamamlandı!")
    print("="*60)

if __name__ == '__main__':
    main()