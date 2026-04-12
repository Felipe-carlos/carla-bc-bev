import torch
import random
import numpy as np

def set_seed(seed=42):
    """
    Garante reprodutibilidade em treinos do PyTorch.
    
    Args:
        seed (int): valor da seed (padrão: 42)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # Para um GPU
        torch.cuda.manual_seed_all(seed)       # Para múltiplos GPUs
    
    # Garantir que operações convolucionais sejam determinísticas
    torch.backends.cudnn.deterministic = True
    
    # Desativar "benchmark" do cuDNN (ele acelera mas escolhe algoritmos não-determinísticos)
    torch.backends.cudnn.benchmark = False