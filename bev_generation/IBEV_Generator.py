from abc import ABC, abstractmethod


class IBEVGenerator(ABC):
    def __init__(self, model_path:str = None, device='cuda', use_eval=True):
        super().__init__()
        self.generator = None

    @abstractmethod
    def infer(self, expert_obs_dict):
        """
        Inferência sem gradiente (eval mode)
        saida deve ser binária (0 ou 1) para cada pixel

        Returns:
            bev tensor (B, C, H, W)
        """
        pass

