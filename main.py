"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, tüm bileşenleri bir araya getiren ANA PROGRAM'dır.
Bir orkestra şefi gibi diğer modülleri sırayla çağırır:

  1. Makroekonomik verileri çeker (FRED)
  2. ETF fiyat verilerini çeker (Yahoo Finance)
  3. Verileri normalize eder
  4. Sektör uzmanlarını eğitir (veya daha önce eğitilmişleri yükler)
  5. Backtest yapar ve sonuçları gösterir
  6. (Opsiyonel) IBKR borsasında canlı işlem yapar

Kullanım:
  python main.py                → Tam çalıştırma (eğitim + backtest)
  python main.py --train        → Sadece modelleri eğit
  python main.py --backtest     → Sadece backtest yap (eğitilmiş modeller gerekir)
  python main.py --live         → Canlı sinyal üret (IBKR bağlantısı gerekir)
"""

import argparse
import sys
import os
import numpy as np

from config import SECTOR_SYMBOLS, EXPERT_FOLDER
from data_fetcher import get_macro_data, prepare_training_data, finalize_state_space
from model_trainer import train_sector_experts, StrategicAgentModule
from backtester import run_backtest, plot_backtest_results


def check_experts_exist(symbols=None, expert_folder=None):
    """Eğitilmiş modellerin dosya sisteminde mevcut olup olmadığını kontrol eder."""
    symbols = symbols or SECTOR_SYMBOLS
    expert_folder = expert_folder or EXPERT_FOLDER

    for sector in symbols:
        if not os.path.exists(f"{expert_folder}/ppo_expert_{sector}.zip"):
            return False
    return True


def fetch_and_prepare_data():
    """Tüm verileri çeker, birleştirir ve normalize eder."""
    print("\n" + "=" * 50)
    print("📡 ADIM 1: Makroekonomik Veriler Çekiliyor")
    print("=" * 50)
    macro_table = get_macro_data()

    if macro_table is None:
        print("❌ Makro veriler çekilemedi. Bot durduruluyor.")
        sys.exit(1)

    print(f"✅ Makro veri boyutu: {macro_table.shape}")

    print("\n" + "=" * 50)
    print("📡 ADIM 2: ETF Verileri Çekiliyor")
    print("=" * 50)
    full_data = prepare_training_data(SECTOR_SYMBOLS, macro_table)
    print(f"✅ Eğitim verisi boyutu: {full_data.shape}")

    print("\n" + "=" * 50)
    print("⚙️ ADIM 3: Durum Uzayı Normalleştiriliyor")
    print("=" * 50)
    final_matrix = finalize_state_space(full_data)
    print("✅ Normalizasyon tamamlandı!")

    return final_matrix


def train(final_matrix):
    """Sektör uzmanlarını eğitir."""
    print("\n" + "=" * 50)
    print("🎓 ADIM 4: Sektör Uzmanları Eğitiliyor")
    print("=" * 50)
    train_sector_experts(final_matrix)


def backtest(final_matrix, sam):
    """Backtest çalıştırır ve sonuçları gösterir."""
    print("\n" + "=" * 50)
    print("📊 ADIM 5: Backtest Başlıyor")
    print("=" * 50)
    dates, marl_history, spy_history = run_backtest(
        SECTOR_SYMBOLS, final_matrix, sam
    )
    plot_backtest_results(dates, marl_history, spy_history)


def live_signals(final_matrix, sam):
    """
    Canlı sinyal üretir ve (opsiyonel olarak) IBKR'de emir gönderir.
    """
    print("\n" + "=" * 50)
    print("🔴 CANLI SİNYAL ÜRETİMİ")
    print("=" * 50)

    # Son gün verilerinden sinyal üret
    current_observations = {}
    for sector in SECTOR_SYMBOLS:
        sector_data = final_matrix[final_matrix['Sector_Ticker'] == sector]
        if not sector_data.empty:
            last_row = sector_data.iloc[-1]
            obs = last_row.drop('Sector_Ticker').values.astype(np.float32)
            current_observations[sector] = obs

    # SAM'den stratejik karar al
    signals = sam.get_strategic_allocation(current_observations)

    print("\n🏛️ SAM (Strategic Agent Module) Kararları:")
    for sector, signal in signals.items():
        status = "AL (BUY)" if signal > 0 else "SAT (SELL)" if signal < 0 else "TUT (HOLD)"
        emoji = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        print(f"  {emoji} {sector}: {status}")

    # IBKR bağlantısını dene (opsiyonel)
    try:
        from execution import ExecutionModule
        exe = ExecutionModule()
        exe.check_positions()
        exe.execute_signals(signals)
        exe.disconnect()
    except Exception as e:
        print(f"\n⚠️ IBKR bağlantısı kurulamadı: {e}")
        print("💡 Sinyaller üretildi ama emir gönderilmedi.")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical MARL Sektör ETF Botu"
    )
    parser.add_argument('--train', action='store_true',
                        help='Sektör uzmanlarını eğit')
    parser.add_argument('--backtest', action='store_true',
                        help='Backtest çalıştır')
    parser.add_argument('--live', action='store_true',
                        help='Canlı sinyal üret')
    parser.add_argument('--all', action='store_true',
                        help='Eğitim + Backtest (varsayılan)')

    args = parser.parse_args()

    # Hiçbir argüman verilmezse varsayılan olarak --all çalışır
    if not (args.train or args.backtest or args.live or args.all):
        args.all = True

    # Verileri her durumda çek
    final_matrix = fetch_and_prepare_data()

    if args.train or args.all:
        train(final_matrix)

    if args.backtest or args.all or args.live:
        # Modellerin mevcut olduğundan emin ol
        if not check_experts_exist():
            print("\n⚠️ Eğitilmiş modeller bulunamadı. Önce eğitim yapılacak.")
            train(final_matrix)

        # SAM'ı başlat
        sam = StrategicAgentModule()

        if args.backtest or args.all:
            backtest(final_matrix, sam)

        if args.live:
            live_signals(final_matrix, sam)

    print("\n🏁 Bot çalışması tamamlandı!")


if __name__ == "__main__":
    main()
