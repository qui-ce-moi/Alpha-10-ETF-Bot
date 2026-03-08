"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, botun tüm ayarlarını tek bir yerde toplar.
Sektör ETF sembolleri, makro ekonomik veri kodları ve diğer
sabit değerler burada tanımlanır. Bu sayede bir ayarı değiştirmek
istediğimizde sadece bu dosyayı düzenlememiz yeterlidir.
"""

import os
from dotenv import load_dotenv

# .env dosyasındaki gizli bilgileri (API anahtarları gibi) güvenle yüklüyoruz.
load_dotenv()

# ============================================================
# API ANAHTARLARI
# ============================================================
# Güvenlik: API anahtarları asla kod içine gömülmez,
# her zaman ortam değişkenlerinden (environment variables) okunur.
FRED_API_KEY = os.getenv("FRED_API_KEY")

# IBKR (Interactive Brokers) Bağlantı Bilgileri
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", 7497))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", 1))

# ============================================================
# SEKTÖR ETF SEMBOLLERİ
# ============================================================
# Farklı sektörleri temsil eden ETF'ler.
# SPDR serisi (XL*) sektörel ETF'leri S&P 500'ün
# alt sektörlerine yatırım yapmamızı sağlar.
SECTOR_SYMBOLS = [
    'XLK',  # Teknoloji
    'XLF',  # Finans
    'XLV',  # Sağlık
    'XLE',  # Enerji
    'XLY',  # Tüketici İhtiyari
    'XLI',  # Sanayi
    'XLC',  # İletişim
    'XLU',  # Kamu Hizmetleri
    'XLP',  # Temel Tüketim
    'XLB',  # Malzeme
]

# ============================================================
# FRED MAKROEKONOMİK VERİ SERİLERİ
# ============================================================
# FRED, ABD Merkez Bankası'nın (Federal Reserve) ekonomik veri
# tabanıdır. Her seri kodu belirli bir ekonomik göstergeyi temsil eder.
MACRO_SERIES = {
    'GDP_Growth': 'A191RL1Q225SBEA',      # GSYİH Büyüme Oranı
    'Core_Inflation': 'PCEPILFE',          # Çekirdek Enflasyon
    'Yield_Curve': 'T10Y2Y',               # Getiri Eğrisi (10Y-2Y farkı)
    'NonFarm_Payroll': 'PAYEMS',           # Tarım Dışı İstihdam
    'Real_GDP': 'GDP',                     # Reel GSYİH
    'Core_PCE': 'PCE',                     # Kişisel Tüketim Harcamaları
    'Treasury_Yield': 'DGS10',             # 10 Yıllık Hazine Getirisi
    'Inflation': 'CPIAUCSL',               # Tüketici Fiyat Endeksi
}

# ============================================================
# MODEL EĞİTİM AYARLARI
# ============================================================
# PPO (Proximal Policy Optimization) modelinin eğitim parametreleri.
PPO_TIMESTEPS = 1000  # Her sektör uzmanının eğitim adım sayısı
EXPERT_FOLDER = "sector_experts"  # Eğitilmiş modellerin kaydedildiği klasör

# ============================================================
# VERİ FİLTRESİ
# ============================================================
DATA_START_DATE = '2010-01-01'  # Makro verilerin başlangıç tarihi

# ============================================================
# BACKTEST AYARLARI
# ============================================================
BACKTEST_DAYS = 100  # Backtest kaç gün üzerinden yapılacak
INITIAL_CAPITAL = 100_000.0  # Başlangıç sermayesi ($)
