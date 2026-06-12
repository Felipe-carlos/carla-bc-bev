"""
Plot agent trajectory arrows over a CARLA top-down map image.

Blue arrows  = normal steps
Red arrows   = step that caused episode termination by infraction

Usage (from repo root):
    python plot_city/plot_traj.py

Edit the CONFIG block below to change input file, town, or output.
"""

import json
import math
import queue
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import carla

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAJ_JSON   = Path('/home/felipe_cds/carla-bc-bev/eval_metrics_infraction/town01/trajectories/real-bev_expert.json')
TOWN_NAME   = 'Town01'
OUTPUT_DIR  = Path('plot_city/plot_traj')

HOST        = 'localhost'
PORT        = 2000

IMAGE_WIDTH  = 1024
IMAGE_HEIGHT = 1024
CAMERA_FOV   = 60
PADDING      = 30   # extra meters around the trajectory bounding box

ARROW_STEP      = 5   # draw 1 normal arrow every N steps (reduces clutter)
ARROW_LENGTH    = 7   # pixels, normal arrow body
ARROW_HEAD      = 4   # pixels, normal arrowhead size

INFRACTION_ARROW_LENGTH = 16
INFRACTION_ARROW_HEAD   = 8

COLOR_NORMAL     = (40, 120, 255)   # blue
COLOR_INFRACTION = (220, 30,  30)   # red
# ─────────────────────────────────────────────────────────────────────────────


class Camera:
    def __init__(self, world, w, h, fov, x, y, z, pitch, yaw):
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=pitch, yaw=yaw),
        )
        self._q = queue.Queue()
        self._actor = world.spawn_actor(bp, transform)
        self._actor.listen(self._q.put)

    def get(self):
        img = None
        while img is None or self._q.qsize() > 0:
            img = self._q.get()
        # copy() is required: frombuffer shares memory with the CARLA image object,
        # which gets freed when the callback returns, causing a segfault on access.
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).copy()
        arr = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
        return arr

    def destroy(self):
        if self._actor.is_alive:
            self._actor.stop()   # stop listener before destroying the actor
            self._actor.destroy()


def set_sync_mode(client, sync):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0
    world.apply_settings(settings)


def world_to_pixel(wx, wy, camera_x, camera_y, mtp, image_width, image_height):
    """Convert CARLA world (x, y) to image pixel (px, py)."""
    px = mtp * (wy - camera_y) + image_width / 2
    py = -mtp * (wx - camera_x) + image_height / 2
    return px, py


def draw_arrow(draw, px, py, yaw_deg, length, head_size, color):
    """
    Draw an arrow centred at (px, py) pointing in the CARLA yaw direction.

    CARLA yaw convention: 0° = +X world, positive = clockwise from above.
    Image convention: +X world maps to -Y pixel, +Y world maps to +X pixel.
    """
    rad = math.radians(yaw_deg)
    # Forward vector in image space
    dx =  math.sin(rad)   # CARLA +Y  → pixel +X
    dy = -math.cos(rad)   # CARLA +X  → pixel -Y

    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return
    dx /= norm
    dy /= norm

    # Arrow body: from base to tip
    tip_x = px + dx * length
    tip_y = py + dy * length
    draw.line([(px, py), (tip_x, tip_y)], fill=color, width=2)

    # Arrowhead triangle
    perp_x, perp_y = -dy, dx
    left_x  = tip_x - dx * head_size + perp_x * head_size * 0.5
    left_y  = tip_y - dy * head_size + perp_y * head_size * 0.5
    right_x = tip_x - dx * head_size - perp_x * head_size * 0.5
    right_y = tip_y - dy * head_size - perp_y * head_size * 0.5
    draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)], fill=color)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAJ_JSON) as f:
        traj = json.load(f)

    xs = [p['x'] for p in traj]
    ys = [p['y'] for p in traj]
    n_episodes  = max(p['episode'] for p in traj) + 1
    n_infractions = sum(1 for p in traj if p['is_infraction_terminal'])
    print(f'Loaded {len(traj)} steps | {n_episodes} episodes | {n_infractions} infractions')

    # Bounding box with padding
    min_x, max_x = min(xs) - PADDING, max(xs) + PADDING
    min_y, max_y = min(ys) - PADDING, max(ys) + PADDING

    camera_x = (min_x + max_x) / 2
    camera_y = (min_y + max_y) / 2
    traj_radius = max(max_x - camera_x, max_y - camera_y)
    camera_z = 1.2 * np.tan(np.radians(90 - CAMERA_FOV / 2)) * traj_radius

    # Metres-to-pixel scale (same formula as gen_maps.py)
    mtp = IMAGE_WIDTH / (2 * math.tan(math.radians(CAMERA_FOV) / 2) * camera_z)

    # ── Connect to CARLA and grab background image ────────────────────────────
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)
    world = client.load_world(TOWN_NAME)
    set_sync_mode(client, True)

    cam = Camera(world, IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV,
                 camera_x, camera_y, camera_z, pitch=-90, yaw=0)
    world.tick()
    bg = cam.get()
    cam.destroy()

    image = Image.fromarray(bg)
    draw  = ImageDraw.Draw(image)

    # ── Draw normal arrows (subsampled) ──────────────────────────────────────
    for i, p in enumerate(traj):
        if p['is_infraction_terminal']:
            continue
        if i % ARROW_STEP != 0:
            continue
        px, py = world_to_pixel(p['x'], p['y'], camera_x, camera_y,
                                mtp, IMAGE_WIDTH, IMAGE_HEIGHT)
        draw_arrow(draw, px, py, p['yaw'], ARROW_LENGTH, ARROW_HEAD, COLOR_NORMAL)

    # ── Draw infraction arrows on top (all, larger) ───────────────────────────
    for p in traj:
        if not p['is_infraction_terminal']:
            continue
        px, py = world_to_pixel(p['x'], p['y'], camera_x, camera_y,
                                mtp, IMAGE_WIDTH, IMAGE_HEIGHT)
        draw_arrow(draw, px, py, p['yaw'],
                   INFRACTION_ARROW_LENGTH, INFRACTION_ARROW_HEAD, COLOR_INFRACTION)

    out_path = OUTPUT_DIR / f'{TRAJ_JSON.stem}.png'
    image.save(out_path.as_posix())
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
