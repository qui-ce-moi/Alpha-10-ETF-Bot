"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, botun Interactive Brokers (IBKR) borsası ile 
gerçek zamanlı iletişim kurmasını sağlar.
IBKR, dünya çapında hisse senedi, ETF ve diğer varlıkların 
alım satımını yapabileceğimiz profesyonel bir borsadır.

Bu modül:
  - IBKR'ye bağlanır
  - SAM'den gelen AL/SAT/TUT sinyallerini gerçek emirlere çevirir
  - Portföy durumunu sorgular

⚠️ ÖNEMLİ GÜVENLİK NOTU:
Canlı işlem (live trading) varsayılan olarak devre dışıdır.
Gerçek para ile işlem yapmak için execute_signals fonksiyonundaki
güvenlik kilidini açmanız gerekir.
"""

import os
from ib_insync import IB, Stock, MarketOrder

from config import IB_HOST, IB_PORT, IB_CLIENT_ID


class ExecutionModule:
    """
    IBKR borsasında gerçek zamanlı emir gönderme modülü.
    """

    def __init__(self, auto_connect=True):
        self.ib = IB()

        if auto_connect:
            self.connect()

    def connect(self):
        """IBKR TWS/Gateway'e bağlan."""
        try:
            self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
            print(f"🔗 IBKR Bağlantısı Kuruldu (Port: {IB_PORT})")
        except Exception as e:
            print(f"❌ IBKR Bağlantı Hatası: {e}")
            print("💡 TWS veya IB Gateway'in açık olduğundan emin olun.")

    def disconnect(self):
        """IBKR bağlantısını kapat."""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("🔌 IBKR bağlantısı kapatıldı.")

    def check_positions(self):
        """Mevcut portföy pozisyonlarını gösterir."""
        positions = self.ib.positions()

        print("\n" + "=" * 50)
        print("📦 MEVCUT PORTFÖY DURUMU (IBKR)")
        print("=" * 50)

        if not positions:
            print("Şu an açık pozisyon bulunmuyor.")
        else:
            for p in positions:
                print(f"🔹 Sektör: {p.contract.symbol:<5} | "
                      f"Adet: {p.position:>6} | "
                      f"Ort. Maliyet: ${p.avgCost:.2f}")

        print("=" * 50)
        return positions

    def execute_signals(self, signals_dict, quantity=10):
        """
        SAM'den gelen sinyalleri gerçek borsada uygular.
        
        Parametreler:
          signals_dict: {sektör: sinyal} (sinyal: -1=SAT, 0=TUT, +1=AL)
          quantity: Her işlem için lot miktarı
          
        ⚠️ GÜVENLİK: Gerçek emir satırları yorum halindedir.
        Canlıya geçmek için aşağıdaki yorum satırlarını kaldırın.
        """
        for sector, signal in signals_dict.items():
            if signal == 0:
                print(f"⏸️ {sector}: TUT (işlem yok)")
                continue

            action = "BUY" if signal > 0 else "SELL"
            contract = Stock(sector, 'SMART', 'USD')

            print(f"📨 {sector}: {action} sinyali alındı "
                  f"({quantity} adet)")

            # ⚠️ GÜVENLİK KİLİDİ: Aşağıdaki satırları aktif etmeden
            # gerçek emir gönderilmez. Test aşamasında güvende kalırız.
            # ---
            # order = MarketOrder(action, quantity)
            # trade = self.ib.placeOrder(contract, order)
            # self.ib.sleep(2)  # Emrin işlenmesi için bekle
            # print(f"✅ {sector}: Emir başarıyla gönderildi!")
            # ---

            print(f"🔒 {sector}: Güvenlik kilidi aktif - emir gönderilmedi.")
