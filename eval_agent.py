import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from bev_generation.unet import Unet_BEVGenerator
from bev_generation.cvt_3ch import CVT_3chL1Generator 
from bev_generation.cvt_6ch import CVT_6chVanilla
from pathlib import Path
import cv2 
import torch as th
from typing import Literal, Optional
from bev_generation.bev_buffer import TemporalBEVBuffer



def create_image_tensor(obs, unet=False, w_resize=192, h_resize=192):
    
    def process_image(image: np.ndarray, traj=False):

        if image.ndim == 4:
            image = image[0]

        if not traj:
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (w_resize, h_resize))
            image = image.transpose(2, 0, 1)
        else:
                 
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))  # CHW → HWC

            image = cv2.resize(image, (w_resize, h_resize), interpolation=cv2.INTER_NEAREST)

            
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))
            else:
                image = image[None, :, :]

        return th.as_tensor(image, dtype=th.float32) / 255.0

    image_tensor_list = []

    if unet:
        traj_plot = process_image(obs['traj_plot'], traj=True)
        camera_order = ["central_rgb", "left_rgb", "right_rgb", "rear_rgb"]
    else:
        camera_order = ["left_rgb", "central_rgb", "right_rgb", "rear_rgb"]
        traj_plot = process_image(obs['traj_plot_rgb'], traj=True)

    for i in camera_order:
        image_tensor_list.append(process_image(obs[i]))

    image_tensor_list.append(traj_plot)
    if unet:
        images = th.cat(image_tensor_list, dim=0)
    else:
        images = th.stack(image_tensor_list, dim=0)

    return images.unsqueeze(0)


def add_label_to_image(img, label, font_scale=0.6, thickness=2, pad=10):
    """Adiciona uma barra com título no topo da imagem."""
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
        
    h, w = img.shape[:2]
    # Cria barra superior
    label_height = max(30, int(h * 0.08))
    labeled_img = np.ones((h + label_height, w, 3), dtype=np.uint8) * 40  # Fundo cinza escuro
    labeled_img[label_height:, :, :] = img
    
    # Texto do label
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x = (w - text_size[0]) // 2
    y = label_height - (label_height - text_size[1]) // 2
    cv2.putText(labeled_img, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled_img


def prepare_image(img, max_h=None, max_w=None):
    """Prepara imagem para composição: tensor->numpy, CHW->HWC, uint8, RGB."""
    if img is None:
        return None
    if isinstance(img, th.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] <= 10:  # CHW -> HWC
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def compose_frame(obs, bev_gen, real_bev, fixed_size=150):
    """
    Monta frame com todas as imagens em tamanho fixo (padrão 150x150):
    - Topo: 4 câmeras RGB (150x150 cada) + labels
    - Base: Real BEV (150x150) | Generated BEV (150x150) | Route (150x150) + labels
    Retorna: np.ndarray (H, W, 3), uint8, RGB
    """
    
    FIXED_SIZE = fixed_size  # Todas as imagens serão FIXED_SIZE x FIXED_SIZE
    
    def resize_to_fixed_square(img, size):
        """Redimensiona qualquer imagem para um quadrado fixo, mantendo conteúdo visível com padding."""
        if img is None:
            return np.zeros((size, size, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        # Resize mantendo aspect ratio dentro do quadrado
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.zeros((size, size, 3), dtype=np.uint8)
        # Centralizar no quadrado com padding cinza
        square = np.zeros((size, size, 3), dtype=np.uint8)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        square[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        return square
    
    # === Preparar câmeras RGB (150x150 cada) ===
    camera_labels = {
        "left_rgb": "Left Cam",
        "central_rgb": "Central Cam", 
        "right_rgb": "Right Cam",
        "rear_rgb": "Rear Cam"
    }
    
    rgb_imgs_labeled = []
    for key in ["left_rgb", "central_rgb", "right_rgb", "rear_rgb"]:
        img = prepare_image(obs.get(key))
        img_fixed = resize_to_fixed_square(img, FIXED_SIZE)
        labeled = add_label_to_image(img_fixed, camera_labels[key])
        rgb_imgs_labeled.append(labeled)
    
    # === Preparar BEVs (150x150 cada) ===
    # Se a BEV for temporal (9 canais), visualizamos apenas os 3 últimos (t) para o vídeo
    real_bev_img = prepare_image(real_bev)
    gen_bev_img = prepare_image(bev_gen)
    
    real_fixed = resize_to_fixed_square(real_bev_img, FIXED_SIZE)
    gen_fixed = resize_to_fixed_square(gen_bev_img, FIXED_SIZE)
    
    real_labeled = add_label_to_image(real_fixed, "Real BEV")
    gen_labeled = add_label_to_image(gen_fixed, "Generated BEV")
    
    # === Preparar trajetória (150x150) ===
    traj_key = 'traj_plot' if 'traj_plot' in obs else 'traj_plot_rgb'
    traj_img = prepare_image(obs.get(traj_key))
    traj_fixed = resize_to_fixed_square(traj_img, FIXED_SIZE)
    traj_labeled = add_label_to_image(traj_fixed, "Route")
    
    # === Montar linha superior (4 câmeras RGB 150x150) ===
    rgb_row = np.hstack(rgb_imgs_labeled)  # 4 x 150 = 600px width (sem label)
    
    # === Montar linha inferior (3 componentes 150x150) ===
    bev_row = np.hstack([real_labeled, gen_labeled, traj_labeled])  # 3 x 150 = 450px width
    
    # === Equalizar larguras das duas linhas ===
    max_w = max(rgb_row.shape[1], bev_row.shape[1])
    if rgb_row.shape[1] != max_w:
        rgb_row = cv2.resize(rgb_row, (max_w, rgb_row.shape[0]), interpolation=cv2.INTER_LINEAR)
    if bev_row.shape[1] != max_w:
        bev_row = cv2.resize(bev_row, (max_w, bev_row.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # === Juntar linhas verticalmente ===
    final_frame = np.vstack([rgb_row, bev_row])
    
    # Garantir formato final para ImageEncoder
    if final_frame.ndim == 2:
        final_frame = cv2.cvtColor(final_frame, cv2.COLOR_GRAY2RGB)
    if final_frame.shape[2] == 4:
        final_frame = final_frame[:, :, :3]
    if final_frame.dtype != np.uint8:
        final_frame = final_frame.astype(np.uint8)
        
    return final_frame  # RGB, uint8, (H, W, 3)

   

def evaluate_policy(
        env, 
        policy,
        video_path,
        min_eval_steps=3000,
        arc:Literal['unet', 'cvt', 'expert','cvt_6ch']='unet',
        video_save_dir: Optional[str] = None, 
        temporal_buffer=False
    ):

    device = 'cuda'
    video_dir = Path(video_save_dir) if video_save_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"🎥 Evaluation video directory set: {video_dir}")

    # Inicialização do Gerador de BEV
    bev_generator = None
    if arc != 'expert':
        output_dir = Path('outputs')
        last_checkpoint_path = output_dir / 'checkpoint.txt'
        if arc == 'unet':
            bev_generator = Unet_BEVGenerator(device=device)
        elif arc == 'cvt':
            bev_generator = CVT_3chL1Generator(device=device)
        elif arc == 'cvt_6ch':
            bev_generator = CVT_6chVanilla(device=device)

    # ==========================================
    # INICIALIZAÇÃO DO BUFFER TEMPORAL
    # ==========================================
    temporal_bev_buffer = None
    if temporal_buffer:
        # Nota: Assumimos height/width padrão de 192 para CVT. 
        # Se usar UNET ou outras resoluções, ajuste aqui conforme necessário.
       
        temporal_bev_buffer = TemporalBEVBuffer(
            device=device, 
            num_envs=env.num_envs, 
            channels=3
        )
        print(f"⏱️  Temporal buffer enabled: [t-2, t-1, t] concatenation")

    policy = policy.eval()
    t0 = time.time()
    for i in range(env.num_envs):
        env.set_attr('eval_mode', True, indices=i)
    obs = env.reset()
    
    # Reset do buffer no início da avaliação
    if temporal_bev_buffer is not None:
        temporal_bev_buffer.reset()
    
    list_render = []
    
    # === Encoder para vídeo BEV+RGB usando ImageEncoder do Gym ===
    bev_video_encoder = None
    bev_video_path = None
    
    ep_stat_buffer = []
    route_completion_buffer = []
    ep_events = {}
    for i in range(env.num_envs):
        ep_events[f'venv_{i}'] = []

    n_step = 0
    n_timeout = 0
    env_done = np.array([False]*env.num_envs)
    
    print(f'Starting evaluation for at least {min_eval_steps} steps or until all environments are done...')
    while n_step < min_eval_steps or not np.all(env_done):
        # Captura a BEV real (ground truth) antes de ser sobrescrita
        real_bev = obs.get('birdview', obs.get('topdown', None))
        
        if arc != 'expert':
            if arc == 'unet':
                unet = True
                image_input = {'image': create_image_tensor(obs,unet=unet).to(device)}
                bev = bev_generator.infer(image_input)
                
            elif arc == 'cvt':
                unet = False
                image_input = {
                    'image': create_image_tensor(obs,unet=unet,w_resize=256,h_resize=256).to(device),
                    'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
                    'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
                }
                bev = bev_generator.infer(image_input)
                
            elif arc == 'cvt_6ch':
                unet = False
                image_input = {
                    'image': create_image_tensor(obs,unet=unet,w_resize=480,h_resize=224).to(device),
                    'extrinsics': th.as_tensor(obs['extrinsics'], dtype=th.float32).to(device),
                    'intrinsics': th.as_tensor(obs['intrinsics'], dtype=th.float32).to(device),
                }
                bev = bev_generator.infer(image_input)
            
            # ==========================================
            # APLICAÇÃO DO BUFFER TEMPORAL
            # ==========================================
            if temporal_buffer and temporal_bev_buffer is not None:
                # Concatena histórico: (B, 3, H, W) -> (B, 9, H, W)
                bev = temporal_bev_buffer.get_concat(bev)
            
            # Atualiza a observação com a BEV (potencialmente temporal)
            obs['birdview'] = bev
                
            # Grava frames para o vídeo BEV+RGB se o diretório foi especificado
            if video_dir is not None:
                # Para visualização, passamos a bev gerada (se for temporal, a função de prepare_image
                # pode precisar de ajuste para visualizar apenas os 3 canais centrais, mas o compose_frame
                # atual lida com tensors genéricos).
                frame = compose_frame(obs, bev, real_bev)  

                if bev_video_encoder is None:
                    bev_video_path = video_dir / "evaluation_bev_rgb.mp4"
                    bev_video_encoder = ImageEncoder(
                        output_path=str(bev_video_path),
                        frame_shape=frame.shape,
                        frames_per_sec=30,
                        output_frames_per_sec=30
                    )
                    print(f"📹 Recording BEV+RGB: {bev_video_path} | Shape: {frame.shape} | @ 30fps")

                if bev_video_encoder is not None:
                    bev_video_encoder.capture_frame(frame)

        # Forward da política (a policy deve esperar 9 canais se temporal_buffer=True)
        actions, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
        obs, reward, done, info = env.step(actions)

        

        for i in range(env.num_envs):
            env.set_attr('action_log_probs', log_probs[i], indices=i)
            env.set_attr('action_mu', mu[i], indices=i)
            env.set_attr('action_sigma', sigma[i], indices=i)

        list_render.append(env.render(mode='rgb_array'))

        n_step += 1
        env_done |= done

        # ==========================================
        # RESET PARCIAL DO BUFFER (PARA EPISÓDIOS QUE TERMINARAM)
        # ==========================================
        if temporal_bev_buffer is not None:
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                temporal_bev_buffer.reset(env_indices=done_indices)

        for i in np.where(done)[0]:
            if not info[i]['timeout']:
                ep_stat_buffer.append(info[i]['episode_stat'])
            if n_step < min_eval_steps or not np.all(env_done):
                route_completion_buffer.append(info[i]['route_completion'])
            ep_events[f'venv_{i}'].append(info[i]['episode_event'])
            n_timeout += int(info[i]['timeout'])

    for ep_info in info:
        route_completion_buffer.append(ep_info['route_completion'])

    # === Salva vídeo padrão do render (já existente) ===
    encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
    for im in list_render:
        encoder.capture_frame(im)
    encoder.close()

    # === Finaliza o encoder do vídeo BEV+RGB ===
    if bev_video_encoder is not None:
        bev_video_encoder.close()
        print(f"✅ BEV/RGB video saved successfully: {bev_video_path}")

    avg_ep_stat = get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
    avg_route_completion = get_avg_route_completion(route_completion_buffer, prefix='eval/')
    avg_ep_stat['eval/eval_timeout'] = n_timeout

    duration = time.time() - t0
    avg_ep_stat['time/t_eval'] = duration
    avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

    for i in range(env.num_envs):
        env.set_attr('eval_mode', False, indices=i)
    obs = env.reset()
    return avg_ep_stat, avg_route_completion, ep_events


def get_avg_ep_stat(ep_stat_buffer, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_stat_buffer))
    if n_episodes > 0:
        for ep_info in ep_stat_buffer:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}completed_n_episodes'] = n_episodes

    return avg_ep_stat


def get_avg_route_completion(ep_route_completion, prefix=''):
    avg_ep_stat = {}
    n_episodes = float(len(ep_route_completion))
    if n_episodes > 0:
        for ep_info in ep_route_completion:
            for k, v in ep_info.items():
                k_avg = f'{prefix}{k}'
                if k_avg in avg_ep_stat:
                    avg_ep_stat[k_avg] += v
                else:
                    avg_ep_stat[k_avg] = v

        for k in avg_ep_stat.keys():
            avg_ep_stat[k] /= n_episodes
    avg_ep_stat[f'{prefix}avg_n_episodes'] = n_episodes

    return avg_ep_stat