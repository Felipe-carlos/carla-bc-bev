import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec # Nova importação para o layout avançado
from PIL import Image

# ==========================================
# --- CONFIGURAÇÕES GERAIS (PARÂMETROS) ---
# ==========================================
PASTA_SAIDA = "Gifs_Dataset"    
PREFIXO_ARQUIVO = "animacao"    

QUADRO_INICIAL = 100             
QTD_QUADROS = 100
# ==========================================

def criar_gif_para_rota(caminho_rota, nome_rota):
    json_path = os.path.join(caminho_rota, "episode.json")
    
    if not os.path.exists(json_path):
        print(f"  [!] Ignorando {nome_rota}: JSON não encontrado.")
        return

    df = pd.read_json(json_path)
    total_frames_disponiveis = len(df)
    
    if QUADRO_INICIAL >= total_frames_disponiveis:
        print(f"  [!] Ignorando {nome_rota}: A corrida tem apenas {total_frames_disponiveis} frames, mas o inicial é {QUADRO_INICIAL}.")
        return

    frames_a_processar = min(QTD_QUADROS, total_frames_disponiveis - QUADRO_INICIAL)
    print(f"  -> Processando {nome_rota}: Do frame {QUADRO_INICIAL} ao {QUADRO_INICIAL + frames_a_processar - 1}...")

    # Criação da Figura mais larga para acomodar as 3 colunas
    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title(f'Gerando {nome_rota}')

    # Define a grade estrutural: 2 linhas, 3 colunas.
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # Associa cada imagem a uma "célula" específica da grade
    ax_left = fig.add_subplot(gs[0, 0])
    ax_central = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    ax_birdview = fig.add_subplot(gs[1, 1]) # Fica na linha de baixo, na coluna do meio
    ax_legend = fig.add_subplot(gs[1, 2])   # Fica na linha de baixo, na coluna da direita

    # Desativa as bordas e eixos de todos os quadros (incluindo o espaço vazio da legenda)
    for ax in [ax_left, ax_central, ax_right, ax_birdview, ax_legend]:
        ax.axis('off')

    # Títulos respeitando a capitalização de primeira palavra
    ax_left.set_title('Câmera esquerda')
    ax_central.set_title('Câmera central')
    ax_right.set_title('Câmera direita')
    ax_birdview.set_title('Birdview (visão superior)')

    # --- CONFIGURAÇÃO DA LEGENDA (NA CÉLULA INFERIOR DIREITA) ---
    legendas = [
        mpatches.Patch(color='black', label='Não trafegável'),
        mpatches.Patch(color='#FF0000', label='Via'),
        mpatches.Patch(color='#FF00FF', label='Limite da via'),
        mpatches.Patch(color='yellow', label='Rota alvo')
    ]
    
    # Anexamos a legenda no eixo vazio que criamos especificamente para ela
    ax_legend.legend(handles=legendas, loc='center left', fontsize='12', framealpha=0.9)
    
    # Ajusta o layout deixando espaço no topo (0.92) para o título principal da figura
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Inicializa os quadros do Matplotlib
    img_left = ax_left.imshow(Image.open(os.path.join(caminho_rota, 'left_rgb', f'{QUADRO_INICIAL:04d}.png')))
    img_central = ax_central.imshow(Image.open(os.path.join(caminho_rota, 'central_rgb', f'{QUADRO_INICIAL:04d}.png')))
    img_right = ax_right.imshow(Image.open(os.path.join(caminho_rota, 'right_rgb', f'{QUADRO_INICIAL:04d}.png')))
    img_birdview = ax_birdview.imshow(Image.open(os.path.join(caminho_rota, 'birdview_masks', f'{QUADRO_INICIAL:04d}_00.png')))

    def atualizar(i):
        frame_idx = QUADRO_INICIAL + i
        acao = df['actions'].iloc[frame_idx]
        
        # Formata o nome da rota e o título respeitando a capitalização exigida
        numero_rota = nome_rota.split('_')[-1]
        fig.suptitle(f"Rota {numero_rota} | Quadro: {frame_idx:03d} | Aceleração/freio: {acao[0]:.2f} | Direção: {acao[1]:.2f}", fontsize=16)
        
        img_left.set_data(Image.open(os.path.join(caminho_rota, 'left_rgb', f'{frame_idx:04d}.png')))
        img_central.set_data(Image.open(os.path.join(caminho_rota, 'central_rgb', f'{frame_idx:04d}.png')))
        img_right.set_data(Image.open(os.path.join(caminho_rota, 'right_rgb', f'{frame_idx:04d}.png')))
        img_birdview.set_data(Image.open(os.path.join(caminho_rota, 'birdview_masks', f'{frame_idx:04d}_00.png')))
        
        return [img_left, img_central, img_right, img_birdview]

    anim = FuncAnimation(fig, atualizar, frames=frames_a_processar, interval=100, blit=False)
    
    nome_arquivo_final = f"{PREFIXO_ARQUIVO}_{nome_rota}.gif"
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo_final)
    
    anim.save(caminho_saida, writer='pillow', fps=10)
    plt.close()

if __name__ == '__main__':
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    pastas_das_rotas = sorted(glob.glob("gail_experts/route_*/ep_00"))
    
    if not pastas_das_rotas:
        print("Nenhuma rota encontrada na pasta gail_experts.")
    else:
        print(f"Iniciando varredura. Encontradas {len(pastas_das_rotas)} rotas para processar.\n")
        
        for caminho in pastas_das_rotas:
            partes_caminho = caminho.split(os.sep) 
            nome_da_rota = partes_caminho[-2] 
            criar_gif_para_rota(caminho_rota=caminho, nome_rota=nome_da_rota)
            
        print(f"\nConcluído! Todos os GIFs foram salvos na pasta '{PASTA_SAIDA}'.")