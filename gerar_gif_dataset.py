import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw
import numpy as np
import torch
import cv2
from bev_generation.unet import Unet_BEVGenerator
from expert_dataset_def.expert_dataset import traj_plotter, traj_plotter_rgb

# ==========================================
# --- CONFIGURAÇÕES GERAIS (PARÂMETROS) ---
# ==========================================
PASTA_SAIDA = "Gifs_Dataset"    
PREFIXO_ARQUIVO = "animacao"    
QUADRO_INICIAL = 100             
QTD_QUADROS = 100
# ==========================================


def create_image_tensor(obs, unet=False, w_resize=192, h_resize=192):
    """
    Prepara tensor de entrada para a UNet.
    Gera o traj_plot internamente a partir das coordenadas brutas em obs['traj'].
    """
    def process_image(image: np.ndarray, traj=False):
        # Garante que estamos trabalhando com numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        if image.ndim == 4:
            image = image[0]  # Remove batch dim se existir

        if not traj:
            # Imagens das câmeras vêm como (H, W, 3) do PIL/numpy
            # Redimensiona primeiro mantendo formato HWC
            image = cv2.resize(image, (w_resize, h_resize),interpolation=cv2.INTER_NEAREST)  # (H, W, 3)
            # Converte para CHW para o tensor PyTorch
            image = image.transpose(2, 0, 1)  # (3, H, W)
        else:
            # traj_plot: pode vir como (H, W) ou (1, H, W)
            if image.ndim == 3 and image.shape[0] == 1:
                image = image[0]  # Remove channel dim extra se existir
            image = cv2.resize(image, (w_resize, h_resize),interpolation=cv2.INTER_NEAREST)  # (H, W)
            image = image[None, :, :]  # (1, H, W)

        return torch.as_tensor(image, dtype=torch.float32) / 255.0

    image_tensor_list = []
    

    # GERAÇÃO DO TRAJ_PLOT DENTRO DA FUNÇÃO
    if unet:
        camera_order = ["central_rgb", "left_rgb", "right_rgb", "rear_rgb"]
        # traj_plotter retorna tensor (1, H, W) em [0, 255]
        traj_tensor = traj_plotter(obs['traj'], w_resize, h_resize)
        traj_plot = traj_tensor / 255.0  # Normaliza para [0, 1]
    else:
        camera_order = ["left_rgb", "central_rgb", "right_rgb", "rear_rgb"]
        # traj_plotter_rgb retorna tensor (3, H, W) em [0, 255]
        traj_tensor = traj_plotter_rgb(obs['traj'], w_resize, h_resize)
        traj_plot = traj_tensor / 255.0
    
    for i in camera_order:
        image_tensor_list.append(process_image(obs[i]))

    image_tensor_list.append(traj_plot)
    images = torch.cat(image_tensor_list, dim=0)  # Concatena ao longo dos canais: 4*3 + 1 = 13
    return images.unsqueeze(0)  # Adiciona batch dim: (1, 13, 192, 192)


def criar_gif_para_rota(caminho_rota, nome_rota):
    json_path = os.path.join(caminho_rota, "episode.json")
    
    if not os.path.exists(json_path):
        print(f"  [!] Ignorando {nome_rota}: JSON não encontrado.")
        return

    df = pd.read_json(json_path)
    total_frames_disponiveis = len(df)
    
    if QUADRO_INICIAL >= total_frames_disponiveis:
        print(f"  [!] Ignorando {nome_rota}: Frames insuficientes.")
        return

    frames_a_processar = min(QTD_QUADROS, total_frames_disponiveis - QUADRO_INICIAL)
    print(f"  -> Processando {nome_rota}: Do frame {QUADRO_INICIAL} ao {QUADRO_INICIAL + frames_a_processar - 1}...")

    # --- CONFIGURAÇÃO DO MODELO UNET ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/felipe_cds/carla-bc-bev/bev_generation/unet/focal_50_generator_49.pth'
    bev_generator = Unet_BEVGenerator(model_path=model_path, device=device)
    
    # --- CONFIGURAÇÃO DO LAYOUT (2x4) ---
    fig = plt.figure(figsize=(18, 8))
    fig.canvas.manager.set_window_title(f'Gerando {nome_rota}')
    
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    ax_left = fig.add_subplot(gs[0, 0])
    ax_central = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    ax_rear = fig.add_subplot(gs[0, 3])
    
    ax_birdview = fig.add_subplot(gs[1, 0])
    ax_generated_bev = fig.add_subplot(gs[1, 1])
    ax_legend = fig.add_subplot(gs[1, 2:4])

    for ax in [ax_left, ax_central, ax_right, ax_rear, ax_birdview, ax_generated_bev, ax_legend]:
        ax.axis('off')

    ax_left.set_title('Câmera esquerda')
    ax_central.set_title('Câmera central')
    ax_right.set_title('Câmera direita')
    ax_rear.set_title('Câmera traseira')
    ax_birdview.set_title('Birdview (Ground Truth)')
    ax_generated_bev.set_title('BEV Gerada (UNet)')

    legendas = [
        mpatches.Patch(color='black', label='Não trafegável'),
        mpatches.Patch(color='#FF0000', label='Via'),
        mpatches.Patch(color='#FF00FF', label='Limite da via'),
        mpatches.Patch(color='yellow', label='Rota alvo')
    ]
    ax_legend.legend(handles=legendas, loc='center left', fontsize='12', framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # --- INICIALIZAÇÃO ---
    base_frame = f'{QUADRO_INICIAL:04d}'
    
    def load_cam(path):
        return np.array(Image.open(path).convert('RGB'))
    
    img_left = ax_left.imshow(load_cam(os.path.join(caminho_rota, 'left_rgb', f'{base_frame}.png')))
    img_central = ax_central.imshow(load_cam(os.path.join(caminho_rota, 'central_rgb', f'{base_frame}.png')))
    img_right = ax_right.imshow(load_cam(os.path.join(caminho_rota, 'right_rgb', f'{base_frame}.png')))
    img_rear = ax_rear.imshow(load_cam(os.path.join(caminho_rota, 'rear_rgb', f'{base_frame}.png')))
    img_birdview = ax_birdview.imshow(load_cam(os.path.join(caminho_rota, 'birdview_masks', f'{base_frame}_00.png')))
    
    # Placeholder para BEV gerada
    img_generated_bev = ax_generated_bev.imshow(np.zeros((192, 192, 3), dtype=np.uint8))

    # --- FUNÇÃO DE ATUALIZAÇÃO ---
    def atualizar(i):
        frame_idx = QUADRO_INICIAL + i
        acao = df['actions'].iloc[frame_idx]
        numero_rota = nome_rota.split('_')[-1]
        
        fig.suptitle(f"Rota {numero_rota} | Quadro: {frame_idx:03d} | Aceleração/freio: {acao[0]:.2f} | Direção: {acao[1]:.2f}", fontsize=16)
        
        # Carrega imagens das câmeras
        obs = {
            'left_rgb': load_cam(os.path.join(caminho_rota, 'left_rgb', f'{frame_idx:04d}.png')),
            'central_rgb': load_cam(os.path.join(caminho_rota, 'central_rgb', f'{frame_idx:04d}.png')),
            'right_rgb': load_cam(os.path.join(caminho_rota, 'right_rgb', f'{frame_idx:04d}.png')),
            'rear_rgb': load_cam(os.path.join(caminho_rota, 'rear_rgb', f'{frame_idx:04d}.png')),
            'traj': df['traj'].iloc[frame_idx],  # Passa coordenadas brutas
        }
        
        # Prepara tensor para inferência e roda a UNet
        inp_tensor = create_image_tensor(obs, unet=True).to(device)
        
          # Economiza memória durante inferência
        bev_pred = bev_generator.infer({'image': inp_tensor})
        
        # Converte saída da UNet para imagem RGB
        bev_np = bev_pred.cpu().numpy().squeeze(0).transpose(1, 2, 0) * 255  # (192, 192, 3) em [0, 255]
        
        # --- GERA TRAJ_PLOT PARA EXIBIÇÃO NO MATPLOTLIB ---
        traj_tensor_display = traj_plotter(obs['traj'], 192, 192)  # (1, 192, 192)
        traj_np = traj_tensor_display.squeeze(0).cpu().numpy()  # (192, 192)
        traj_rgb_display = np.stack([traj_np] * 3, axis=-1)  # (192, 192, 3) para exibição RGB
        
        # Atualiza todos os elementos do plot
        img_left.set_data(load_cam(os.path.join(caminho_rota, 'left_rgb', f'{frame_idx:04d}.png')))
        img_central.set_data(load_cam(os.path.join(caminho_rota, 'central_rgb', f'{frame_idx:04d}.png')))
        img_right.set_data(load_cam(os.path.join(caminho_rota, 'right_rgb', f'{frame_idx:04d}.png')))
        img_rear.set_data(load_cam(os.path.join(caminho_rota, 'rear_rgb', f'{frame_idx:04d}.png')))
        img_birdview.set_data(load_cam(os.path.join(caminho_rota, 'birdview_masks', f'{frame_idx:04d}_00.png')))
        img_generated_bev.set_data(bev_np.astype(np.uint8))
        
        return [img_left, img_central, img_right, img_rear, img_birdview, img_generated_bev]

    anim = FuncAnimation(fig, atualizar, frames=frames_a_processar, interval=100, blit=False)
    
    nome_arquivo_final = f"{PREFIXO_ARQUIVO}_{nome_rota}.gif"
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo_final)
    anim.save(caminho_saida, writer='pillow', fps=10)
    plt.close()

if __name__ == '__main__':
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    pastas_das_rotas = sorted(glob.glob("expert-data/route_*/ep_00"))
    
    if not pastas_das_rotas:
        print("Nenhuma rota encontrada na pasta expert-data.")
    else:
        print(f"Iniciando varredura. Encontradas {len(pastas_das_rotas)} rotas para processar.\n")
        for caminho in pastas_das_rotas:
            nome_da_rota = caminho.split(os.sep)[-2] 
            criar_gif_para_rota(caminho_rota=caminho, nome_rota=nome_da_rota)
        print(f"\nConcluído! GIFs salvos em '{PASTA_SAIDA}'.")