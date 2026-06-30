import sys
import queue
from pathlib import Path

import glob
import os

# Raiz do projeto (cvt-6ch) para encontrar carla_gym
sys.path.append("/home/felipe_cds/carla-bc-bev")


import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import carla
from carla import ColorConverter as cc

from carla_gym.utils import config_utils
from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from carla_gym.core.task_actor.common.navigation.route_manipulation import downsample_route


# --- Classe Camera ---
class Camera(object):
    def __init__(self, world, w, h, fov, x, y, z, pitch, yaw):
        bp_library = world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(w))
        camera_bp.set_attribute('image_size_y', str(h))
        camera_bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.queue = queue.Queue()
        self.camera = world.spawn_actor(camera_bp, transform)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None
        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def __del__(self):
        pass


def set_sync_mode(client, sync):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0
    world.apply_settings(settings)


# --- Configuração ---
# Pasta mãe do dataset (ex: /caminho/para/town01)
parent_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/felipe_cds/cvt-6ch/plot_city/only_episodes_json')
town_name = 'Town01'  # Nome do mapa CARLA correspondente

# Cores por dataset (BGR para PIL = RGB)
DATASET_COLORS = [
    (220,  50,  50),   # Vermelho
    ( 50, 200,  50),   # Verde
    ( 50,  50, 220),   # Azul
    (220, 160,   0),   # Laranja
    (160,  50, 220),   # Roxo
]

# Descobre subpastas de dataset em ordem
dataset_dirs = sorted([d for d in parent_dir.iterdir() if d.is_dir()])
if not dataset_dirs:
    raise RuntimeError(f"Nenhuma subpasta encontrada em {parent_dir}")
print(f"Datasets encontrados ({len(dataset_dirs)}):")
for i, d in enumerate(dataset_dirs):
    print(f"  [{i}] {d.name}  cor={DATASET_COLORS[i % len(DATASET_COLORS)]}")

# Para cada dataset, descobre quais route_XX existem
def get_route_indices(dataset_dir: Path):
    indices = []
    for p in sorted(dataset_dir.iterdir()):
        if p.is_dir() and p.name.startswith('route_'):
            try:
                indices.append(int(p.name.split('_')[1]))
            except ValueError:
                pass
    return indices


# --- Conecta ao CARLA ---
host = 'localhost'
port = 2000
client = carla.Client(host, port)
client.set_timeout(30.0)
world = client.get_world()
set_sync_mode(client, True)

map = world.get_map()
planner = GlobalRoutePlanner(map, resolution=1.0)

traj_output_dir = Path(f'plot_routes/{town_name}')
traj_output_dir.mkdir(parents=True, exist_ok=True)

routes_file = Path(f'carla_gym/envs/scenario_descriptions/LeaderBoard/{town_name}/routes.xml')
route_descriptions_dict = config_utils.parse_routes_file(routes_file)
route_ids = sorted(route_descriptions_dict.keys())

# --- Planejar todas as rotas e calcular bounding box global ---
planned_routes = {}   # route_id → list of waypoints
min_x = float('inf')
max_x = -float('inf')
min_y = float('inf')
max_y = -float('inf')

for route_id in route_ids:
    route_description = route_descriptions_dict[route_id]
    route_config = route_description['ego_vehicles']
    spawn_transform = route_config['hero'][0]
    target_transforms = route_config['hero'][1:]
    current_location = spawn_transform.location

    global_plan_world_coord = []
    for tt in target_transforms:
        next_target_location = tt.location
        route_trace = planner.trace_route(current_location, next_target_location)
        global_plan_world_coord += route_trace
        current_location = next_target_location

    planned_routes[route_id] = global_plan_world_coord

    for trajectory_point in global_plan_world_coord:
        point_loc = trajectory_point[0].transform.location
        if point_loc.x < min_x: min_x = point_loc.x
        if point_loc.x > max_x: max_x = point_loc.x
        if point_loc.y < min_y: min_y = point_loc.y
        if point_loc.y > max_y: max_y = point_loc.y

# --- Câmera top-down ---
camera_x = (max_x + min_x) / 2
camera_y = (max_y + min_y) / 2
traj_radius = max(max_x - camera_x, max_y - camera_y)

image_width = 512
image_height = 512
camera_fov = 60
camera_z = 1.2 * np.tan((90 - camera_fov / 2) * np.pi / 180) * traj_radius

camera = Camera(world, image_width, image_height, camera_fov,
                camera_x, camera_y, camera_z, -90, 0)
world.tick()
result = camera.get()
image = Image.fromarray(result)
draw = ImageDraw.Draw(image)

meters_to_pixel_x = image_width  / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
meters_to_pixel_y = -image_height / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)


def world_to_pixel(wx, wy):
    """Converte coordenadas mundiais CARLA (x, y) para pixel na imagem."""
    px = meters_to_pixel_x * (wy - camera_y) + image_width  / 2
    py = meters_to_pixel_y * (wx - camera_x) + image_height / 2
    return px, py


def get_spawn_xy(route_id):
    """Retorna (x, y) do ponto de spawn da rota a partir do routes.xml."""
    spawn_loc = route_descriptions_dict[route_id]['ego_vehicles']['hero'][0].location
    return spawn_loc.x, spawn_loc.y


def integrate_trajectory(ep_json_path, spawn_x, spawn_y, dt=0.1):
    """Integra vel_xy do episode.json para reconstruir a trajetória em coords mundiais."""
    df = pd.read_json(ep_json_path)
    positions = [(spawn_x, spawn_y)]
    x, y = spawn_x, spawn_y
    for i in range(len(df)):
        vx, vy = df.iloc[i]['vel_xy']
        x += vx * dt
        y += vy * dt
        positions.append((x, y))
    return positions


def draw_trajectory(draw, positions, color, width=2, step=5):
    """Desenha a trajetória; usa step para reduzir pontos e acelerar o plot."""
    sampled = positions[::step]
    if len(sampled) < 2:
        return
    lx, ly = world_to_pixel(*sampled[0])
    for wx, wy in sampled[1:]:
        px, py = world_to_pixel(wx, wy)
        draw.line((lx, ly, px, py), width=width, fill=color)
        lx, ly = px, py


# --- Desenhar trajetórias reais por dataset (integração de vel_xy) ---
for ds_idx, ds_dir in enumerate(dataset_dirs):
    color = DATASET_COLORS[ds_idx % len(DATASET_COLORS)]
    route_indices = get_route_indices(ds_dir)
    drawn = 0
    for idx in route_indices:
        route_id = route_ids[idx] if idx < len(route_ids) else None
        if route_id is None or route_id not in planned_routes:
            continue

        ep_json = ds_dir / f'route_{idx:02d}' / 'ep_00' / 'episode.json'
        if not ep_json.exists():
            print(f"  [aviso] {ep_json} não encontrado, pulando")
            continue

        spawn_x, spawn_y = get_spawn_xy(route_id)
        positions = integrate_trajectory(ep_json, spawn_x, spawn_y)
        draw_trajectory(draw, positions, color, width=2)
        drawn += 1

    print(f"{ds_dir.name}: {drawn} trajetórias desenhadas na cor {color}")

# --- Legenda ---
legend_x = 8
legend_y = 8
box_size = 12
line_height = 16
padding = 4

# Fundo semitransparente (PIL não suporta alpha em Draw, então desenhamos um retângulo escuro)
legend_w = 180
legend_h = line_height * len(dataset_dirs) + padding * 2
draw.rectangle([legend_x - padding, legend_y - padding,
                legend_x + legend_w, legend_y + legend_h],
               fill=(0, 0, 0))

for ds_idx, ds_dir in enumerate(dataset_dirs):
    color = DATASET_COLORS[ds_idx % len(DATASET_COLORS)]
    y = legend_y + ds_idx * line_height
    draw.rectangle([legend_x, y + 2, legend_x + box_size, y + box_size], fill=color)
    draw.text((legend_x + box_size + 4, y), ds_dir.name, fill=(255, 255, 255))

# --- Salvar ---
image_path = traj_output_dir / f'{town_name}_all_datasets.png'
image.save(image_path.as_posix())
print(f"Imagem salva em: {image_path}")
