from pathlib import Path
from matplotlib.cm import get_cmap
import queue
from pathlib import Path

from PIL import Image, ImageDraw, ImageColor
import numpy as np

import carla
from carla import ColorConverter as cc

from carla_gym.utils import config_utils
from carla_gym.core.task_actor.common.navigation.global_route_planner import GlobalRoutePlanner
from carla_gym.core.task_actor.common.navigation.route_manipulation import downsample_route

# --- Classe Camera (não alterada) ---
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

# --- Conecta ao CARLA ---
host = 'localhost'
port = 2002
client = carla.Client(host, port)
client.set_timeout(30.0)
world = client.load_world('Town02')
set_sync_mode(client, True)

map = world.get_map()
planner = GlobalRoutePlanner(map, resolution=1.0)

traj_output_dir = Path('plot_routes/Town2')
traj_output_dir.mkdir(parents=True, exist_ok=True)

routes_file = Path('carla_gym/envs/scenario_descriptions/LeaderBoard/Town02/routes.xml')
route_descriptions_dict = config_utils.parse_routes_file(routes_file)

# --- 1. Processar todas as rotas para calcular o bounding box global ---
all_global_plan_world_coord = []
all_ds_ids = []

min_x = float('inf')
max_x = -float('inf')
min_y = float('inf')
max_y = -float('inf')

for route_id, route_description in route_descriptions_dict.items():
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

    ds_ids = downsample_route(global_plan_world_coord, 50)
    all_global_plan_world_coord.append(global_plan_world_coord)
    all_ds_ids.append(ds_ids)

    # Atualiza o bounding box global
    for trajectory_point in global_plan_world_coord:
        point_loc = trajectory_point[0].transform.location
        if point_loc.x < min_x: min_x = point_loc.x
        if point_loc.x > max_x: max_x = point_loc.x
        if point_loc.y < min_y: min_y = point_loc.y
        if point_loc.y > max_y: max_y = point_loc.y

# --- 2. Posicionar câmera no centro do bounding box global ---
camera_x = (max_x + min_x) / 2
camera_y = (max_y + min_y) / 2
traj_radius = max(max_x - camera_x, max_y - camera_y)

image_width = 512
image_height = 512
camera_fov = 60
camera_z = 1.2 * np.tan((90 - camera_fov / 2) * np.pi / 180) * traj_radius
camera = Camera(world, image_width, image_height, camera_fov, camera_x, camera_y, camera_z, -90, 0)
world.tick()
result = camera.get()
image = Image.fromarray(result)
draw = ImageDraw.Draw(image)


# --- 3. Desenhar rotas na imagem (rota 1 por último para garantir visibilidade) ---
meters_to_pixel_x = image_width / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
meters_to_pixel_y = -image_height / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)

# # Definir cores
COLOR_ROUTE_1 = (0, 0, 255)  # Azul
COLOR_OTHER_ROUTES = (200, 0, 0)  # Cinza

# # --- 1. Desenhar todas as rotas EXCETO a rota 1 ---
# for route_idx, global_plan_world_coord in enumerate(all_global_plan_world_coord):
#     if route_idx == 0:
#         continue  # Pula a rota 1, ela será desenhada depois

#     # Desenhar rotas em cinza
#     last_point = global_plan_world_coord[0][0].transform.location
#     last_point_x = meters_to_pixel_x * (last_point.y - camera_y) + image_width / 2
#     last_point_y = meters_to_pixel_y * (last_point.x - camera_x) + image_height / 2

#     for point_idx in range(1, len(global_plan_world_coord)):
#         point_loc = global_plan_world_coord[point_idx][0].transform.location
#         point_x = meters_to_pixel_x * (point_loc.y - camera_y) + image_width / 2
#         point_y = meters_to_pixel_y * (point_loc.x - camera_x) + image_height / 2

#         draw.line((last_point_x, last_point_y, point_x, point_y), width=2, fill=COLOR_OTHER_ROUTES)
#         last_point_x = point_x
#         last_point_y = point_y

# # --- 2. Desenhar a rota 1 por último (em azul) ---
# route_idx = 0
# global_plan_world_coord = all_global_plan_world_coord[route_idx]

# last_point = global_plan_world_coord[0][0].transform.location
# last_point_x = meters_to_pixel_x * (last_point.y - camera_y) + image_width / 2
# last_point_y = meters_to_pixel_y * (last_point.x - camera_x) + image_height / 2

# radius = 3
# # Destacar ponto inicial da rota 1
# draw.ellipse([last_point_x - radius*2, last_point_y - radius*2, 
#              last_point_x + radius*2, last_point_y + radius*2], 
#              fill=COLOR_ROUTE_1)

# for point_idx in range(1, len(global_plan_world_coord)):
#     point_loc = global_plan_world_coord[point_idx][0].transform.location
#     point_x = meters_to_pixel_x * (point_loc.y - camera_y) + image_width / 2
#     point_y = meters_to_pixel_y * (point_loc.x - camera_x) + image_height / 2

#     draw.line((last_point_x, last_point_y, point_x, point_y), width=2, fill=COLOR_ROUTE_1)
#     last_point_x = point_x
#     last_point_y = point_y

# rect_size = 5  # Tamanho do retângulo (ajuste conforme necessário)
# rect_coords = [
#     last_point_x - rect_size,  # Esquerda
#     last_point_y - rect_size,  # Topo
#     last_point_x + rect_size,  # Direita
#     last_point_y + rect_size   # Base
# ]
# draw.rectangle(rect_coords, outline=COLOR_ROUTE_1, width=5) 

import numpy as np
from scipy.interpolate import interp1d

# --- Extrair pontos da rota 1 ---
route_idx = 0
global_plan_world_coord = all_global_plan_world_coord[route_idx]

# Extrair coordenadas (x, y) da rota
xs = [p[0].transform.location.x for p in global_plan_world_coord]
ys = [p[0].transform.location.y for p in global_plan_world_coord]

# --- Interpolar para 2000 pontos igualmente espaçados ---
num_points = 2654

# Calcular distância acumulada entre pontos
dist = np.cumsum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2))
dist = np.insert(dist, 0, 0)  # Adicionar início

# Criar função de interpolação com base na distância
interp_x = interp1d(dist, xs, kind='linear')
interp_y = interp1d(dist, ys, kind='linear')

# Gerar pontos igualmente espaçados
dist_fine = np.linspace(0, dist[-1], num_points)
xs_fine = interp_x(dist_fine)
ys_fine = interp_y(dist_fine)

pedacos = {
    'partida': (0, 33),
    'reta': [(34, 115), (236, 615), (731, 815), (926, 950), (1076, 1185), (1310, 1320)],
    'curva à direita': [(116, 235), (616, 730)],
    'cruzamento à direita': (816, 925),
    'cruzamento reto': (951, 1075),
    'cruzamento à esquerda': [(1186, 1309), (1321, 1440)]
}

cores = {
    'partida': 'orange',
    'reta': 'blue',
    'curva à direita': 'green',
    'cruzamento à direita': 'purple',
    'cruzamento reto': 'red',
    'cruzamento à esquerda': 'brown'
}

# --- Função auxiliar para verificar se um índice pertence a algum intervalo ---
def get_color_for_index(idx, pedacos):
    for tipo, intervalos in pedacos.items():
        if isinstance(intervalos, list):  # Para listas de intervalos (ex: 'reta', 'curva à direita', etc.)
            for inicio, fim in intervalos:
                if inicio <= idx <= fim:
                    return tipo
        else:  # Para intervalos únicos (ex: 'partida', 'cruzamento à direita', etc.)
            inicio, fim = intervalos
            if inicio <= idx <= fim:
                return tipo
    return None  # Fora dos intervalos definidos

# --- Desenhar rota com cores por segmento ---
for i in range(1, len(xs_fine)):
    idx_atual = i  # Índice do ponto interpolado (de 0 a 1999)

    # Verificar cor do ponto atual
    tipo_atual = get_color_for_index(idx_atual, pedacos)
    if tipo_atual is None:
        continue  # Pular pontos fora dos intervalos definidos

    # Converter coordenadas para pixels
    x0 = meters_to_pixel_x * (ys_fine[i - 1] - camera_y) + image_width / 2
    y0 = meters_to_pixel_y * (xs_fine[i - 1] - camera_x) + image_height / 2
    x1 = meters_to_pixel_x * (ys_fine[i] - camera_y) + image_width / 2
    y1 = meters_to_pixel_y * (xs_fine[i] - camera_x) + image_height / 2

    # Aplicar cor correspondente
    cor = cores.get(tipo_atual, 'gray')  # Padrão: cinza (não deve acontecer)
    draw.line((x0, y0, x1, y1), fill=cor, width=4)

# --- Destacar início e fim ---
x_inicio = meters_to_pixel_x * (ys_fine[0] - camera_y) + image_width / 2
y_inicio = meters_to_pixel_y * (xs_fine[0] - camera_x) + image_height / 2
draw.ellipse([x_inicio - 5, y_inicio - 5, x_inicio + 5, y_inicio + 5], fill='orange')  # Início




# --- 4. Salvar imagem única com todas as rotas ---
image_path = traj_output_dir / 'T2_route1.png'
image.save(image_path.as_posix())
print(f"Imagem salva em: {image_path}")