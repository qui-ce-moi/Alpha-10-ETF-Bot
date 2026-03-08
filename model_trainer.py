"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, yapay zekâ modellerini eğitmek ve yönetmekten sorumludur.
İki ana iş yapar:
  1. Eğitim (Training): Her sektör ETF'si için ayrı bir PPO modeli 
     eğitir ve kaydeder. PPO (Proximal Policy Optimization) bir 
     pekiştirmeli öğrenme algoritmasıdır.
  2. SAM (Strategic Agent Module): Eğitilmiş tüm sektör uzmanlarını 
     yükleyip, her birine "bugünkü piyasa durumuna göre ne yapmalıyız?"
     diye sorar. Uzmanların cevaplarını toplar ve bir strateji oluşturur.
"""

import os
import numpy as np
from stable_baselines3 import PPO

from config import SECTOR_SYMBOLS, EXPERT_FOLDER, PPO_TIMESTEPS
from environment import SectorTradingEnv


def train_sector_experts(final_matrix, symbols=None, expert_folder=None, timesteps=None):
    """
    Her sektör için ayrı bir PPO uzman modeli eğitir ve diske kaydeder.
    
    Parametreler:
      final_matrix: Normalize edilmiş eğitim verisi
      symbols: Eğitilecek sektör sembolleri
      expert_folder: Modellerin kaydedileceği klasör
      timesteps: Her modelin kaç adım eğitileceği
    """
    symbols = symbols or SECTOR_SYMBOLS
    expert_folder = expert_folder or EXPERT_FOLDER
    timesteps = timesteps or PPO_TIMESTEPS

    # Klasör yoksa oluştur
    os.makedirs(expert_folder, exist_ok=True)

    for sector in symbols:
        print(f"🎓 {sector} uzmanı eğitiliyor ({timesteps} adım)...")
        sector_data = final_matrix[final_matrix['Sector_Ticker'] == sector]

        if sector_data.empty:
            print(f"⚠️ {sector} için veri bulunamadı, atlanıyor.")
            continue

        env = SectorTradingEnv(sector_data)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)
        model.save(f"{expert_folder}/ppo_expert_{sector}")
        print(f"✅ {sector} uzmanı kaydedildi.")

    print("\n🎉 Tüm sektör uzmanları eğitimi tamamlandı!")


class StrategicAgentModule:
    """
    Stratejik Ajan Modülü (SAM):
    Tüm sektör uzmanlarını yükler ve her birinden piyasa görüşü alarak
    bir yatırım stratejisi oluşturur.
    
    Bir "general kurul" gibi düşünebilirsiniz:
    Her sektör uzmanı kendi alanındaki görüşünü bildirir,
    SAM ise bu görüşleri birleştirerek nihai kararı verir.
    """

    def __init__(self, symbols=None, expert_folder=None):
        self.symbols = symbols or SECTOR_SYMBOLS
        self.experts = {}

        expert_folder = expert_folder or EXPERT_FOLDER

        # Tüm sektör uzmanlarını (PPO modellerini) yükle
        print("🗂️ Sektör uzmanları yükleniyor...")
        for sector in self.symbols:
            model_path = f"{expert_folder}/ppo_expert_{sector}"
            if os.path.exists(model_path + ".zip"):
                self.experts[sector] = PPO.load(model_path)
                print(f"  ✅ {sector} uzmanı yüklendi.")
            else:
                print(f"  ⚠️ {sector} uzmanı bulunamadı: {model_path}")
        print(f"✅ Toplam {len(self.experts)} uzman masada!")

    def get_strategic_allocation(self, current_market_state_dict):
        """
        Her uzmana kendi sektörünün güncel verilerini gösterir ve
        AL/SAT/TUT sinyali alır.
        
        Döndürdüğü değer:
          {sektör_sembolü: sinyal} şeklinde bir sözlük
          sinyal: -1 (SAT), 0 (TUT), +1 (AL)
        """
        expert_signals = {}

        for sector in self.symbols:
            if sector not in self.experts:
                continue

            # Her uzmana sadece kendi sektörünün verisini ver
            obs = current_market_state_dict[sector]
            action, _ = self.experts[sector].predict(obs, deterministic=True)

            # Eylemleri sinyale çevir: 0=SAT(-1), 1=TUT(0), 2=AL(+1)
            signal = action - 1
            expert_signals[sector] = signal

        return expert_signals
