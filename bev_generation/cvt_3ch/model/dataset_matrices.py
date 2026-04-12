import numpy as np

def deg2rad(d):
    return d * np.pi / 180.0

def euler_to_R(roll_deg, pitch_deg, yaw_deg):
    r, p, y = map(deg2rad, (roll_deg, pitch_deg, yaw_deg))
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return Rz @ Ry @ Rx

def extrinsic_cam(cam_location, cam_rotation, ego_x_px, ego_y_px, bev_w, bev_h, pixels_per_meter):
    """
    Calcula a matriz extrínseca 4x4 de uma única câmera no referencial BEV centrado.

    Args:
        cam_location: [x,y,z] da câmera em metros, relativo ao ego.
        cam_rotation: [roll,pitch,yaw] da câmera em graus.
        ego_x_px, ego_y_px: posição do ego em pixels na BEV.
        bev_w, bev_h: largura e altura da BEV em pixels.
        pixels_per_meter: escala px/m.
    
    Returns:
        T: matriz extrínseca 4x4 (camera -> BEV centrado)
    """
    # centro da BEV
    origin_x, origin_y = bev_w // 2, bev_h // 2

    # posição do ego em metros no referencial centrado
    ego_dx_px = ego_x_px - origin_x
    ego_dy_px = ego_y_px - origin_y
    ego_x_m = -ego_dy_px / pixels_per_meter  # frente
    ego_y_m = ego_dx_px / pixels_per_meter   # direita
    ego_pos_m = np.array([ego_x_m, ego_y_m, 0.0])

    # posição absoluta da câmera
    cam_pos_m = ego_pos_m + np.array(cam_location)

    # rotação
    R = euler_to_R(*cam_rotation)

    # matriz extrínseca 4x4
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = cam_pos_m

    return T

def extrinsic_bev(ego_px, ego_py, bev_w, bev_h, pixels_per_meter):
    """
    Cria uma câmera BEV virtual sobre o ego (não no centro), olhando para baixo.
    ego_px, ego_py: posição do ego em pixels
    """
    origin_x, origin_y = bev_w // 2, bev_h // 2

    # posição do ego em metros no sistema centrado
    ego_x_m = -(ego_py - origin_y)/pixels_per_meter
    ego_y_m = (ego_px - origin_x)/pixels_per_meter
    ego_pos_m = np.array([ego_x_m, ego_y_m, 0.0])

    # área da BEV em metros
    area_x = bev_h / pixels_per_meter
    area_y = bev_w / pixels_per_meter

    # distância máxima do ego até os cantos da BEV
    max_x = max(abs(-area_x/2 - ego_x_m), abs(area_x/2 - ego_x_m))
    max_y = max(abs(-area_y/2 - ego_y_m), abs(area_y/2 - ego_y_m))
    diag = np.sqrt(max_x**2 + max_y**2)

    h = diag / 2  # altura da câmera

    # posição absoluta da câmera
    cam_pos = ego_pos_m + np.array([0.0, 0.0, h])

    # rotação para olhar para baixo
    roll, pitch, yaw = 0.0, -90.0, 0.0

    # matriz extrínseca
    R = euler_to_R(roll, pitch, yaw)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = cam_pos

    return T

def intrinsic_bev(bev_w, bev_h, pixels_per_meter):
    u0 = bev_w / 2
    v0 = bev_h / 2
    K = np.array([
        [pixels_per_meter, 0, u0],
        [0, pixels_per_meter, v0],
        [0, 0, 1]
    ])
    return K

def intrinsic_cam(fov,width,height):
    """
    Calcula a matriz intrínseca a partir dos parâmetros da câmera.
    :param camera_params: Dicionário com os parâmetros da câmera.
    :return: Matriz intrínseca 3x3.
    """
    
    # Converter FOV para radianos
    fov_rad = np.radians(fov)
    
    # Calcular distâncias focais (fx, fy) em pixels
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    
    # O ponto principal (cx, cy) é assumido como o centro da imagem
    cx = width / 2
    cy = height / 2
    
    # Matriz intrínseca K
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    
    return K

import numpy as np
import torch
from matplotlib.path import Path as mb_path

# ----------------- Máscaras -----------------
def get_camera_masks(
    H_bev=32, 
    W_bev=32, 
    H_img=14, 
    W_img=30, 
    n_cameras=None,    
    batch=4, 
    n_heads=4,
    device='cuda'
):
    pixels_per_meter = 5.0
    ego_x, ego_y = 96, 152
    bev_w, bev_h = 192, 192  
    mask_size = H_bev        

    # ---- Definição das câmeras ----
    cameras = {
        'left_rgb':   {'location': [1.2, -0.25, 1.3], 'rotation': [0.0, 0.0, -45.0], 'fov': 90},
        'central_rgb':{'location': [1.2,  0.00, 1.3], 'rotation': [0.0, 0.0,   0.0], 'fov': 90},
        'right_rgb':  {'location': [1.2,  0.25, 1.3], 'rotation': [0.0, 0.0,  45.0], 'fov': 90},
        'rear_rgb':   {'location': [-1.5, 0.00, 1.3], 'rotation': [0.0, 0.0, 180.0], 'fov': 90},
        'traj':       {'location': [0.0,  0.00, 0.0], 'rotation': [0.0, 0.0,   0.0], 'fov': 360}
    }

    # Ordem fixa
    # camera_order = ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'traj']
    camera_order = ['right_rgb', 'rear_rgb', 'left_rgb', 'central_rgb', 'traj']
    
    if n_cameras is None:
        n_cameras = len(camera_order)

    # ---- Auxiliares ----
    def world_to_bev(x, y):
        px = ego_x + (y * pixels_per_meter)
        py = ego_y - (x * pixels_per_meter)
        return px, py

    def clip_polygon_against_rect(poly, W, H):
        def _intersect_seg_with_vertical(p1, p2, xk):
            (x1, y1), (x2, y2) = p1, p2
            if abs(x2 - x1) < 1e-12: return None
            t = (xk - x1) / (x2 - x1)
            if 0 <= t <= 1: return (xk, y1 + t * (y2 - y1))
            return None
        def _intersect_seg_with_horizontal(p1, p2, yk):
            (x1, y1), (x2, y2) = p1, p2
            if abs(y2 - y1) < 1e-12: return None
            t = (yk - y1) / (y2 - y1)
            if 0 <= t <= 1: return (x1 + t * (x2 - x1), yk)
            return None
        def clip_edge(vertices, edge):
            out = []
            if not vertices: return out
            for i in range(len(vertices)):
                curr, prev = vertices[i], vertices[i-1]
                if edge == 'left':
                    inside_curr, inside_prev = curr[0] >= 0, prev[0] >= 0
                elif edge == 'right':
                    inside_curr, inside_prev = curr[0] <= (W-1), prev[0] <= (W-1)
                elif edge == 'top':
                    inside_curr, inside_prev = curr[1] >= 0, prev[1] >= 0
                elif edge == 'bottom':
                    inside_curr, inside_prev = curr[1] <= (H-1), prev[1] <= (H-1)
                if inside_curr:
                    if not inside_prev:
                        ip = (_intersect_seg_with_vertical(prev, curr, 0) if edge=='left'
                              else _intersect_seg_with_vertical(prev, curr, W-1) if edge=='right'
                              else _intersect_seg_with_horizontal(prev, curr, 0) if edge=='top'
                              else _intersect_seg_with_horizontal(prev, curr, H-1))
                        if ip is not None: out.append(ip)
                    out.append(curr)
                elif inside_prev:
                    ip = (_intersect_seg_with_vertical(prev, curr, 0) if edge=='left'
                          else _intersect_seg_with_vertical(prev, curr, W-1) if edge=='right'
                          else _intersect_seg_with_horizontal(prev, curr, 0) if edge=='top'
                          else _intersect_seg_with_horizontal(prev, curr, H-1))
                    if ip is not None: out.append(ip)
            return out
        verts = [tuple(v) for v in poly]
        for e in ('left','right','top','bottom'):
            verts = clip_edge(verts, e)
            if not verts: return np.zeros((0,2))
        return np.array(verts)

    def camera_fov_polygon(cam, bev_w, bev_h, far_distance=200.0):
        x, y, _ = cam['location']
        yaw = deg2rad(cam['rotation'][2])
        fov = deg2rad(cam['fov']) / 2.0
        left_yaw, right_yaw = yaw - fov, yaw + fov
        left_x, left_y = x + far_distance * np.cos(left_yaw), y + far_distance * np.sin(left_yaw)
        right_x, right_y = x + far_distance * np.cos(right_yaw), y + far_distance * np.sin(right_yaw)
        p_cam = world_to_bev(x, y)
        p_left = world_to_bev(left_x, left_y)
        p_right = world_to_bev(right_x, right_y)
        poly = [p_cam, p_left, p_right]
        return clip_polygon_against_rect(poly, bev_w, bev_h)

    # ---- Construção das máscaras ----
    masks = []
    cell_w, cell_h = bev_w/mask_size, bev_h/mask_size

    for name in camera_order:
        cam = cameras[name]
        if name == "traj":
            mask = torch.zeros((mask_size, mask_size), dtype=torch.float32, device=device)
        else:
            poly = camera_fov_polygon(cam, bev_w, bev_h)
            path = mb_path(poly)
            mask = np.ones((mask_size, mask_size), dtype=np.float32)
            for i in range(mask_size):
                for j in range(mask_size):
                    cx, cy = (j+0.5) * cell_w, (i+0.5) * cell_h
                    if path.contains_point((cx, cy)):
                        mask[i, j] = 0
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            mask = torch.where(mask == 0, torch.zeros_like(mask), torch.full_like(mask, float('-inf')))
        masks.append(mask)

    masks_stacked = torch.stack(masks, dim=0)                 
    masks_bev = masks_stacked.view(n_cameras, -1).t()         

    total_pixels_per_cam = H_img * W_img
    masks_expanded = masks_bev.unsqueeze(-1).expand(-1, -1, total_pixels_per_cam)
    mask_att = masks_expanded.reshape(H_bev*W_bev, n_cameras*total_pixels_per_cam)
    mask_att = mask_att.unsqueeze(0).repeat(batch*n_heads, 1, 1)

    return mask_att
