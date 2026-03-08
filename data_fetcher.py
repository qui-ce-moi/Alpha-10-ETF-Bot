"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, botun ihtiyaç duyduğu tüm verileri internetten çeker.
İki tür veri çeker:
  1. Makroekonomik Veriler: FRED'den (ABD Merkez Bankası veritabanı) 
     GSYİH, enflasyon, istihdam gibi ekonomik göstergeler.
  2. ETF Fiyat Verileri: Yahoo Finance'den sektörel ETF fiyat ve hacim bilgileri.
Ayrıca teknik göstergeler (SMA, RSI, Volatilite) hesaplayarak
tüm verileri birleştirir.
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import yfinance as yf

from config import (
    FRED_API_KEY, MACRO_SERIES, SECTOR_SYMBOLS,
    DATA_START_DATE
)


def get_macro_data():
    """
    FRED API'den makroekonomik verileri çeker.
    Tüm serileri tek bir DataFrame'de birleştirir ve 
    eksik değerleri ileri doldurma (forward fill) yöntemiyle tamamlar.
    """
    if not FRED_API_KEY:
        print("❌ Hata: FRED_API_KEY bulunamadı!")
        return None

    fred = Fred(FRED_API_KEY)
    all_series = []

    for name, series_id in MACRO_SERIES.items():
        try:
            s_data = fred.get_series(series_id)
            df_temp = pd.DataFrame(s_data, columns=[name])
            all_series.append(df_temp)
            print(f"✅ {name} çekildi.")
        except Exception as e:
            print(f"❌ {name} ({series_id}) çekilirken hata: {e}")

    if all_series:
        # Tüm serileri tarihe göre birleştir ve sırala
        macro_df = pd.concat(all_series, axis=1).sort_index()
        # 2010 sonrası verileri al ve eksik değerleri doldur
        macro_df = macro_df.loc[DATA_START_DATE:].ffill()
        return macro_df

    return None


def prepare_training_data(symbols, macro_df):
    """
    Her sektör ETF'si için:
      1. Yahoo Finance'den fiyat verisi çeker
      2. Teknik göstergeler hesaplar (SMA-50, SMA-200, RSI, Volatilite)
      3. Makroekonomik verilerle birleştirir
      4. Eksik verileri temizler
    Sonuç olarak tüm sektörlerin verilerini tek bir DataFrame'de döndürür.
    """
    final_dataset = []

    for symbol in symbols:
        print(f"🔄 {symbol} işleniyor...")
        # 1. ETF fiyat verisini çek
        df = yf.download(symbol, start=macro_df.index.min(), progress=False)

        # yfinance'den gelen çoklu sütun seviyeli başlıkları düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Teknik Göstergeler
        # SMA (Basit Hareketli Ortalama): Fiyat trendini gösterir
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # RSI (Göreceli Güç Endeksi): Aşırı alım/satım durumunu gösterir
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # Volatilite: Son 21 günlük fiyat değişimlerinin standart sapması
        df['Volatility'] = df['Close'].pct_change().rolling(window=21).std()

        # 3. Makro verileri birleştir
        merged = df.join(macro_df, how='left')

        # 4. Eksik değerleri doldur ve temizle
        merged = merged.ffill()
        merged = merged.dropna()

        # 5. Sektör etiketini ekle
        merged['Sector_Ticker'] = symbol

        # 6. Final veri kümesine ekle
        final_dataset.append(merged)

    # 7. Tüm sektörlerin verilerini birleştir
    return pd.concat(final_dataset)


def finalize_state_space(full_data):
    """
    Modelin kullanacağı durum uzayını (state space) hazırlar:
      1. Rastgele duygu skoru (sentiment) ekler 
         (gerçek bir FinBERT modeli entegre edilene kadar geçici çözüm)
      2. Tüm sayısal değerleri 0-1 arasına normalize eder
         (Min-Max Scaling). Bunun sebebi: Fiyat 200$ iken enflasyon 0.02
         olursa model kafası karışır.
    """
    np.random.seed(42)
    # Geçici olarak rastgele duygu skorları (-1 ile 1 arası)
    full_data['Sentiment_Score'] = np.random.uniform(-1, 1, size=len(full_data))

    # Sayısal sütunları seç
    numeric_cols = full_data.select_dtypes(include=[np.number]).columns

    # Min-Max Normalizasyon (0-1 arasına çekme)
    full_data[numeric_cols] = (
        (full_data[numeric_cols] - full_data[numeric_cols].min()) /
        (full_data[numeric_cols].max() - full_data[numeric_cols].min())
    )

    return full_data


if __name__ == "__main__":
    # Test: Sadece veri çekme işlemini sına
    print("--- Makro Veriler Çekiliyor ---")
    macro = get_macro_data()
    if macro is not None:
        print(f"\n✅ Makro veri boyutu: {macro.shape}")
        print(macro.tail(5))

        print("\n--- ETF Verileri Hazırlanıyor ---")
        data = prepare_training_data(SECTOR_SYMBOLS, macro)
        print(f"\n✅ Eğitim verisi boyutu: {data.shape}")

        print("\n--- Durum Uzayı Normalleştiriliyor ---")
        final = finalize_state_space(data)
        print("✅ Normalizasyon tamamlandı!")
        print(final[['Close', 'RSI', 'Inflation', 'Sentiment_Score']].tail(5))
