import torch as th
from pathlib import Path



# ==========================================
# CLASSE PARA MANIPULAÇÃO TEMPORAL DA BEV
# ==========================================
class TemporalBEVBuffer:
    def __init__(self, device, channels=3, height=192, width=192):
        self.device = device
        self.channels, self.height, self.width = channels, height, width
        # Inicializa os 2 últimos timestamps com zeros
        self.history = [
            th.zeros(self.channels, self.height, self.width, device=device),
            th.zeros(self.channels, self.height, self.width, device=device)
        ]

    def reset(self):
        """Reseta o buffer para zeros (útil no início de epochs/episódios)"""
        for i in range(len(self.history)):
            self.history[i].zero_()

    def get_concat(self, current_bev):
        """
        Retorna a BEV concatenada: [t-2, t-1, t]
        current_bev: (B, C, H, W)
        Retorna: (B, 3*C, H, W)
        """
        B = current_bev.shape[0]
        # Expande o histórico para o tamanho do batch atual
        h2 = self.history[1].unsqueeze(0).expand(B, -1, -1, -1)
        h1 = self.history[0].unsqueeze(0).expand(B, -1, -1, -1)
        
        # Concatena ao longo da dimensão de canais
        concat_bev = th.cat([h2, h1, current_bev], dim=1)

        # Atualiza o histórico
        # OBS: Para consistência temporal perfeita, o DataLoader deve fornecer dados sequenciais.
        # Aqui usamos o primeiro sample do batch como representação para o próximo passo.
        self.history[1] = self.history[0].clone()
        self.history[0] = current_bev[0].clone()
        
        return concat_bev
