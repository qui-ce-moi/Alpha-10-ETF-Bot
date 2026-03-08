"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, botun geçmiş veri üzerinde "keşke o tarihlerde çalışsaydı
nasıl sonuç alırdık?" sorusunu cevaplar. Buna "Backtest" denir.

100 günlük bir pencere içinde:
  - Her gün SAM'den sinyal alır
  - Ertesi gün o sinyale göre alım/satım yapar
  - S&P 500 (SPY) endeksi ile performansı kıyaslar
  - Sonuçları grafikle gösterir

Bu sayede modelin gerçek para ile kullanılmadan önce
ne kadar başarılı olduğunu test ederiz.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from config import SECTOR_SYMBOLS, BACKTEST_DAYS, INITIAL_CAPITAL


def run_backtest(symbols, full_data, sam_module, 
                 backtest_days=None, initial_capital=None):
    """
    SAM stratejisini geçmiş veriler üzerinde test eder 
    ve S&P 500 ile karşılaştırır.
    
    Parametreler:
      symbols: Sektör ETF sembolleri
      full_data: Tüm eğitim verisi (normalize edilmiş)
      sam_module: Stratejik Ajan Modülü
      backtest_days: Kaç günlük test
      initial_capital: Başlangıç sermayesi
    """
    backtest_days = backtest_days or BACKTEST_DAYS
    initial_capital = initial_capital or INITIAL_CAPITAL

    # 1. Backtest dönemi: Son N günlük tarihler
    test_dates = sorted(full_data.index.unique())[-backtest_days:]
    print(f"🚀 {backtest_days} Günlük Backtest Başlıyor: "
          f"{test_dates[0].date()} -> {test_dates[-1].date()}")

    # 2. Benchmark (S&P 500) verisini çek
    print("📥 S&P 500 (SPY) verisi indiriliyor...")
    spy_data = yf.download(
        'SPY',
        start=test_dates[0],
        end=test_dates[-1] + pd.Timedelta(days=5),
        progress=False
    )

    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_data.columns = spy_data.columns.get_level_values(0)

    spy_data = spy_data.reindex(test_dates).ffill()
    spy_returns = spy_data['Close'].pct_change().fillna(0)

    # 3. Portföy takip değişkenleri
    marl_capital = initial_capital
    spy_capital = initial_capital

    marl_history = [marl_capital]
    spy_history = [spy_capital]

    # Önceki günün sinyallerini sakla (bakış yanlılığını önlemek için)
    prev_signals = {sector: 0 for sector in symbols}

    for i, date in enumerate(test_dates):
        daily_marl_returns = []
        current_signals = {}

        for sector in symbols:
            sector_data = full_data[full_data['Sector_Ticker'] == sector]

            if date in sector_data.index:
                day_row = sector_data.loc[[date]]

                # A: BUGÜNKÜ GETİRİ (Dünkü sinyale göre)
                if i > 0:
                    prev_date = test_dates[i - 1]
                    if prev_date in sector_data.index:
                        today_price = day_row['Close'].values[0]
                        prev_price = sector_data.loc[[prev_date], 'Close'].values[0]
                        ret = (today_price - prev_price) / prev_price

                        signal = prev_signals[sector]
                        if signal == 1:    # AL kararı → uzun pozisyon getirisi
                            daily_marl_returns.append(ret)
                        elif signal == -1: # SAT kararı → kısa pozisyon getirisi
                            daily_marl_returns.append(-ret)
                        else:              # TUT kararı → getiri yok
                            daily_marl_returns.append(0)

                # B: YARINKİ KARAR (SAM'den yeni sinyal al)
                obs = day_row.drop('Sector_Ticker', axis=1).values[0].astype(np.float32)
                action, _ = sam_module.experts[sector].predict(obs, deterministic=True)
                current_signals[sector] = action - 1

        # Sinyalleri bir sonraki güne aktar
        prev_signals = current_signals

        # C: GÜN SONU KAR/ZARAR HESAPLAMASI
        if i > 0:
            # MARL Portföyü (10 sektöre eşit dağılım varsayımı)
            avg_marl_return = np.mean(daily_marl_returns) if daily_marl_returns else 0
            marl_capital *= (1 + avg_marl_return)
            marl_history.append(marl_capital)

            # SPY (Benchmark) Getirisi
            spy_ret = spy_returns.loc[date]
            if isinstance(spy_ret, pd.Series):
                spy_ret = spy_ret.values[0]
            spy_capital *= (1 + float(spy_ret))
            spy_history.append(spy_capital)

    return test_dates, marl_history, spy_history


def plot_backtest_results(dates, marl_history, spy_history, initial_capital=None):
    """
    Backtest sonuçlarını grafik olarak gösterir ve 
    performans raporunu yazdırır.
    """
    initial_capital = initial_capital or INITIAL_CAPITAL

    plt.figure(figsize=(14, 7))

    # MARL Portföyü (Bizim Strateji)
    plt.plot(dates, marl_history,
             label='Hierarchical MARL Strategy',
             color='royalblue', lw=2.5)

    # S&P 500 (Benchmark)
    plt.plot(dates, spy_history,
             label='S&P 500 (SPY)',
             color='dimgray', lw=2, linestyle='--')

    # Başlangıç sermayesi çizgisi
    plt.axhline(y=initial_capital, color='red', linestyle=':', alpha=0.5)

    plt.title('Backtest: MARL vs S&P 500', fontsize=14)
    plt.ylabel('Portföy Değeri ($)', fontsize=12)
    plt.xlabel('Tarih', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("backtest_result.png", dpi=150)
    print("📊 Grafik kaydedildi: backtest_result.png")
    plt.show()

    # Performans Raporu
    marl_perf = ((marl_history[-1] / initial_capital) - 1) * 100
    spy_perf = ((spy_history[-1] / initial_capital) - 1) * 100

    print("\n" + "=" * 40)
    print("📊 Performans Raporu")
    print("=" * 40)
    print(f"Son Tarih          : {dates[-1].date()}")
    print(f"Başlangıç Sermaye  : ${initial_capital:,.2f}")
    print(f"MARL Portföy Sonu  : ${marl_history[-1]:,.2f} (%{marl_perf:.2f})")
    print(f"S&P 500 (SPY) Sonu : ${spy_history[-1]:,.2f} (%{spy_perf:.2f})")
    print(f"Alpha              : %{(marl_perf - spy_perf):.2f}")
    print("=" * 40)
