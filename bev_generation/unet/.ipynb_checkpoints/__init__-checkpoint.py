from bev_generation.IBEV_Generator import IBEVGenerator
from .unet_def import GeneratorUNet
import torch 
import os

class Unet_BEVGenerator(IBEVGenerator):
    def __init__(self, model_path:str=None ,device='cuda', use_eval=True):
        if model_path is None:
            main = os.getcwd()
            model_path = os.path.join(main, 'bev_generation/unet/focal_50_generator_49.pth')
        else:
            model_path = model_path
        in_channels = 13 #4 imagens rgb e 1 de trajetoria e comando
        self.generator = GeneratorUNet(in_channels=in_channels)
        self.generator = self.generator.to(device)
         #-------carrega modelo------------
        
        checkpoint_path = model_path
        state_dict = torch.load(checkpoint_path)

        # Verifique se as chaves no state_dict são compatíveis
        # Se necessário, remova prefixos ou ajuste as chaves conforme necessário
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("generator.", "")  
            new_state_dict[new_key] = value

        # Carregar o state_dict no modelo
        self.generator.load_state_dict(new_state_dict)

        if use_eval:
            self.generator.eval()
    
    def __name__(self):
        return "unet"
        
    def infer(self, expert_obs_dict):
        """
        Inferência sem gradiente (eval mode)

        Returns:
            bev tensor (B, C, H, W)
        """
        with torch.no_grad():
            # Gerar imagem fake (com U-Net)
            fake_birdview = self.generator(expert_obs_dict['image']) 
            fake_birdview = (fake_birdview > 0.5).byte()  # Binarizar a saída
            fake_birdview = fake_birdview *255 # Converter para 0 e 255
        return fake_birdview

