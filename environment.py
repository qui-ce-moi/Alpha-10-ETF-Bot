"""
AÇIKLAMA (Yazılım Bilmeyenler İçin):
Bu dosya, yapay zekâ ajanının "öğrenme ortamını" (environment) tanımlar.
Gymnasium kütüphanesini kullanarak, ajan sanki bir oyun oynuyormuş gibi
piyasa verilerini gözlemler ve 3 eylemden birini seçer:
  - 0: SAT (Sell)
  - 1: TUT (Hold)  
  - 2: AL (Buy)

Her adımda ajan bir ödül (reward) alır:
  - Al dedi + fiyat yükseldi → pozitif ödül
  - Al dedi + fiyat düştü → negatif ödül
  - Sat dedi → tersine çalışır
Bu sayede ajan zaman içinde hangi durumlarda alım/satım
yapması gerektiğini "öğrenir".
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SectorTradingEnv(gym.Env):
    """
    Sektör ETF'leri için pekiştirmeli öğrenme (RL) ortamı.
    Ajan, her gün piyasa verilerine bakarak AL/SAT/TUT kararı verir.
    """

    def __init__(self, full_data):
        super(SectorTradingEnv, self).__init__()

        # Veri boşsa hata fırlat - modele boş veri vermek mantıksız
        if full_data.empty:
            raise ValueError("Hata: Ajana verilen veri seti boş!")

        self.full_data = full_data.reset_index(drop=True)
        self.current_step = 0

        # Eylem Uzayı: 3 ayrı seçenek (SAT=0, TUT=1, AL=2)
        self.action_space = spaces.Discrete(3)

        # Gözlem Uzayı: Sektör adı sütunu sayısal olmadığı için çıkarılır
        self.drop_col = 'Sector_Ticker' if 'Sector_Ticker' in full_data.columns else None
        obs_shape_col_count = len(full_data.columns) - (1 if self.drop_col else 0)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_shape_col_count,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Ortamı başa al - yeni bir eğitim döngüsü başlatır."""
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        """Mevcut günün verilerini gözlem olarak döndürür."""
        row = self.full_data.iloc[self.current_step]

        if self.drop_col:
            obs = row.drop(self.drop_col).values
        else:
            obs = row.values

        return obs.astype(np.float32)

    def step(self, action):
        """
        Ajanın eylemini uygular ve ödülü hesaplar.
        Ödül mantığı: Günlük getiri x ajanın yön kararı
        """
        self.current_step += 1

        price_today = self.full_data.iloc[self.current_step]['Close']
        price_yesterday = self.full_data.iloc[self.current_step - 1]['Close']

        # Sıfıra bölme hatasını önlemek için kontrol
        if price_yesterday == 0:
            daily_return = 0
        else:
            daily_return = (price_today - price_yesterday) / price_yesterday

        # Ödül hesaplaması
        if action == 2:    # AL → fiyat yükselirse ödül pozitif
            reward = daily_return
        elif action == 0:  # SAT → fiyat düşerse ödül pozitif (short pozisyon)
            reward = -daily_return
        else:              # TUT → ödül yok
            reward = 0

        # Son güne ulaşıldıysa oyun biter
        done = self.current_step >= len(self.full_data) - 1
        obs = self._get_observation()

        return obs, reward, done, False, {}
