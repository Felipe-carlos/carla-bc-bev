"""
Plot all model trajectories and infraction arrows over a CARLA top-down map.

One output image per town. Each model gets a distinct color.
- Thin dots   = normal trajectory (every TRAJ_STEP steps)
- Arrow       = step that terminated the episode by infraction

Usage (from repo root):
    python plot_city/plot_all_traj.py /path/to/eval_metrics_infraction
    python plot_city/plot_all_traj.py  # uses DEFAULT_METRICS_DIR
"""

import argparse
import json
import math
import queue
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import carla

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_METRICS_DIR = Path('eval_metrics_spawns')
OUTPUT_DIR          = Path('plot_city/plot_traj')

# Which trajectory files to include (file stem = filename without .json).
# Set to None to include all files found in trajectories/.
MODELS = [
    'cvt_3ch_L1_cvt',
    'real-bev_unet',
    'real-bev_expert',
    'real-bev_cvt_6ch_vanilla_no_noise',
    'real-bev_cvt_6ch',
    'real-bev_cvt_6ch_kde',
]

# Optional display names shown in the legend (stem → label).
# Any stem not listed here falls back to the stem itself.
MODEL_LABELS = {
    'cvt_3ch_L1_cvt':                  'CVT 3 ch.',
    'real-bev_unet':                   'UNet',
    'real-bev_expert':                 'Expert',
    'real-bev_cvt_6ch_vanilla_no_noise': 'CVT 6 ch. no noise ',
    'real-bev_cvt_6ch':                'CVT 6 ch.',
    'real-bev_cvt_6ch_kde':            'CVT 6 ch. KDE',
}

HOST = 'localhost'
PORT = 2000

IMAGE_WIDTH  = 1024
IMAGE_HEIGHT = 1024
CAMERA_FOV          = 60
PADDING             = 20    # metres around bounding box
CAMERA_HEIGHT_FACTOR = 1  # lower = more zoomed in (1.0 = exact fit, 1.2 = 20% margin)

DRAW_SCALE = 1.6   # global multiplier for all drawing sizes (arrows, dots, markers)

TRAJ_STEP       = 10   # draw one trajectory dot every N steps

# Base sizes — multiplied by DRAW_SCALE at runtime
_DOT_RADIUS       = 2
_ARROW_LENGTH     = 14
_ARROW_HEAD       = 12
_START_DOT_RADIUS = 8
_EPISODE_FONT_SZ  = 30  # balloon label font size (also multiplied by DRAW_SCALE)

LEGEND_ITEM_H   = 100   # pixels per legend row
LEGEND_SWATCH   = 60   # colour swatch size
LEGEND_FONT_SZ  = 45   # legend font size

# Fixed colour per model stem — add new stems here as needed
MODEL_COLORS = {
    'cvt_3ch_L1_cvt':                   (220,  50,  50),  # red
    'real-bev_unet':                     ( 50, 130, 255),  # blue
    'real-bev_expert':                   ( 50, 190,  70),  # green
    'real-bev_cvt_6ch_vanilla_no_noise': (255, 160,   0),  # orange
    'real-bev_cvt_6ch':                  (170,  60, 220),  # purple
    'real-bev_cvt_6ch_kde':              (255,  90, 170),  # pink 
    'real-bev_cvt_6ch_traj':             (  0, 155, 110),  # teal
    'real-bev_cvt_6ch_traj_cmd':         (210, 185,   0),  # yellow
    'real-bev_cvt_6ch_traj_cmd_kde':     (255, 130,  50),  # amber
    'cvt_6ch_vanilla_cvt_6ch':           ( 90,  90, 200),  # indigo
    'unet_unet':                         (140,  90,  30),  # brown
}
_FALLBACK_PALETTE = [
    (190, 190,   0), (  0, 180, 180), (180,   0, 180),
]
# ─────────────────────────────────────────────────────────────────────────────


# ── CARLA helpers ─────────────────────────────────────────────────────────────

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
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).copy()
        arr = arr.reshape((img.height, img.width, 4))[:, :, :3][:, :, ::-1]
        return arr

    def destroy(self):
        if self._actor.is_alive:
            self._actor.stop()
            self._actor.destroy()


def set_sync_mode(client, sync):
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = sync
    s.fixed_delta_seconds = 1.0 / 10.0
    world.apply_settings(s)


def capture_background(client, town_name, camera_x, camera_y, camera_z):
    world = client.load_world(town_name)
    set_sync_mode(client, True)
    cam = Camera(world, IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV,
                 camera_x, camera_y, camera_z, pitch=-90, yaw=0)
    world.tick()
    bg = cam.get()
    cam.destroy()
    return bg


# ── geometry helpers ──────────────────────────────────────────────────────────

def world_to_pixel(wx, wy, camera_x, camera_y, mtp):
    px = mtp * (wy - camera_y) + IMAGE_WIDTH  / 2
    py = -mtp * (wx - camera_x) + IMAGE_HEIGHT / 2
    return px, py


def draw_arrow(draw, px, py, yaw_deg, length, head_size, color):
    rad = math.radians(yaw_deg)
    dx =  math.sin(rad)
    dy = -math.cos(rad)
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return
    dx /= norm
    dy /= norm

    tip_x = px + dx * length
    tip_y = py + dy * length
    draw.line([(px, py), (tip_x, tip_y)], fill=color, width=2)

    perp_x, perp_y = -dy, dx
    lx = tip_x - dx * head_size + perp_x * head_size * 0.5
    ly = tip_y - dy * head_size + perp_y * head_size * 0.5
    rx = tip_x - dx * head_size - perp_x * head_size * 0.5
    ry = tip_y - dy * head_size - perp_y * head_size * 0.5
    draw.polygon([(tip_x, tip_y), (lx, ly), (rx, ry)], fill=color)


def draw_dot(draw, px, py, radius, color):
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)


def draw_episode_label(draw, px, py, episode_num, marker_r, font, right=True):
    """Draw the episode number in white above the marker. right=True → above-right, False → above-left."""
    label = str(episode_num)
    try:
        bb = font.getbbox(label)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        tw, th = font.getsize(label)
    by = int(py) - marker_r - th - 2
    bx = int(px) + marker_r + 2 if right else int(px) - marker_r - tw - 2
    draw.text((bx, by), label, fill=(255, 255, 255), font=font)


# ── episode shape markers ─────────────────────────────────────────────────────

# One shape per episode index (cycles if more episodes than shapes)
MARKER_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'pentagon', 'star', 'hexagon', 'triangle_down']


def _poly_pts(cx, cy, r, n, rot_deg=0):
    """Regular polygon vertices."""
    return [
        (cx + r * math.cos(math.radians(rot_deg + i * 360 / n)),
         cy + r * math.sin(math.radians(rot_deg + i * 360 / n)))
        for i in range(n)
    ]


def _star_pts(cx, cy, r_outer, r_inner, n=5, rot_deg=-90):
    pts = []
    for i in range(n * 2):
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(math.radians(rot_deg + i * 180 / n)),
                    cy + r * math.sin(math.radians(rot_deg + i * 180 / n))))
    return pts


def draw_marker(draw, px, py, r, fill_color, outline_color, episode_idx):
    """Draw a shape whose type is determined by episode_idx, with outline then fill."""
    shape = MARKER_SHAPES[episode_idx % len(MARKER_SHAPES)]
    ro = r + 2  # outline radius

    if shape == 'circle':
        draw.ellipse([px-ro, py-ro, px+ro, py+ro], fill=outline_color)
        draw.ellipse([px-r,  py-r,  px+r,  py+r ], fill=fill_color)
    elif shape == 'square':
        draw.rectangle([px-ro, py-ro, px+ro, py+ro], fill=outline_color)
        draw.rectangle([px-r,  py-r,  px+r,  py+r ], fill=fill_color)
    elif shape == 'triangle':
        draw.polygon(_poly_pts(px, py, ro, 3, rot_deg=-90), fill=outline_color)
        draw.polygon(_poly_pts(px, py, r,  3, rot_deg=-90), fill=fill_color)
    elif shape == 'diamond':
        draw.polygon(_poly_pts(px, py, ro, 4, rot_deg=0),   fill=outline_color)
        draw.polygon(_poly_pts(px, py, r,  4, rot_deg=0),   fill=fill_color)
    elif shape == 'pentagon':
        draw.polygon(_poly_pts(px, py, ro, 5, rot_deg=-90), fill=outline_color)
        draw.polygon(_poly_pts(px, py, r,  5, rot_deg=-90), fill=fill_color)
    elif shape == 'star':
        draw.polygon(_star_pts(px, py, ro, ro*0.45), fill=outline_color)
        draw.polygon(_star_pts(px, py, r,  r *0.45), fill=fill_color)
    elif shape == 'hexagon':
        draw.polygon(_poly_pts(px, py, ro, 6, rot_deg=0),   fill=outline_color)
        draw.polygon(_poly_pts(px, py, r,  6, rot_deg=0),   fill=fill_color)
    elif shape == 'triangle_down':
        draw.polygon(_poly_pts(px, py, ro, 3, rot_deg=90),  fill=outline_color)
        draw.polygon(_poly_pts(px, py, r,  3, rot_deg=90),  fill=fill_color)


# ── legend ────────────────────────────────────────────────────────────────────

def _load_serif_font(size):
    candidates = [
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        '/usr/share/fonts/truetype/linux-libertine/LinLibertineR.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_size(font, text):
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return font.getsize(text)


def make_side_legend(height):
    """Vertical strip explaining marker shapes: right side of the map."""
    font = _load_serif_font(LEGEND_FONT_SZ)
    r = LEGEND_SWATCH // 2        # marker radius for illustration
    pad = 24
    item_gap = LEGEND_ITEM_H + 10

    labels = ['Início de rota', 'Final de rota', 'Infração']
    max_tw = max(_text_size(font, lbl)[0] for lbl in labels)
    side_w = pad + 2*r + 4 + 12 + max_tw + pad

    strip = Image.new('RGBA', (side_w, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)

    items = [
        # (fill, outline, draw_arrow_flag, label)
        ((0,   0,   0),   (255, 255, 255), False, 'Início de rota'),
        ((255, 255, 255), (0,   0,   0),   False, 'Final de rota'),
        (None,            None,            True,  'Infração'),
    ]

    y0 = (height - len(items) * item_gap) // 2

    for idx, (fill, outline, is_arrow, label) in enumerate(items):
        cy = y0 + idx * item_gap + r
        cx = pad + r + 2
        if is_arrow:
            # draw a representative arrow in white
            draw_arrow(d, cx - r, cy, 0, 2 * r, r, (220, 220, 220))
        else:
            ro = r + 2
            d.ellipse([cx-ro, cy-ro, cx+ro, cy+ro], fill=outline)
            d.ellipse([cx-r,  cy-r,  cx+r,  cy+r],  fill=fill)
        tw, th = _text_size(font, label)
        d.text((cx + r + 14, cy - th // 2), label, fill=(220, 220, 220), font=font)

    return strip, side_w


def add_legend(image, model_names, colors):
    """Append a legend strip below and a marker-meaning strip to the right."""
    font = _load_serif_font(LEGEND_FONT_SZ)

    # ── bottom strip (model colours) ─────────────────────────────────────────
    n = len(model_names)
    cols = 3
    rows = math.ceil(n / cols)
    strip_h = rows * LEGEND_ITEM_H + 12

    col_label_w = [0] * cols
    for i, name in enumerate(model_names):
        tw, _ = _text_size(font, name)
        col_label_w[i % cols] = max(col_label_w[i % cols], tw)

    item_w = [10 + LEGEND_SWATCH + 8 + col_label_w[c] + 20 for c in range(cols)]
    col_w  = item_w[:]

    # ── side strip (marker meanings) ─────────────────────────────────────────
    side_strip, side_w = make_side_legend(IMAGE_HEIGHT)

    total_w  = max(IMAGE_WIDTH + side_w, sum(item_w))
    strip_w  = total_w

    strip = Image.new('RGBA', (strip_w, strip_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)

    col_x = [sum(col_w[:c]) for c in range(cols)]
    for i, (name, color) in enumerate(zip(model_names, colors)):
        row = i // cols
        col = i  % cols
        x0  = col_x[col] + 10
        y0  = row * LEGEND_ITEM_H + 8
        d.rectangle([x0, y0, x0 + LEGEND_SWATCH, y0 + LEGEND_SWATCH], fill=color)
        draw_arrow(d, x0 + LEGEND_SWATCH // 2, y0 + LEGEND_SWATCH // 2,
                   0, LEGEND_SWATCH // 2, 4, (255, 255, 255))
        d.text((x0 + LEGEND_SWATCH + 8, y0 + 2), name, fill=(220, 220, 220), font=font)

    # ── make map borders transparent ─────────────────────────────────────────
    img_rgba = image.convert('RGBA')
    arr = np.array(img_rgba)
    black_mask = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)
    arr[black_mask, 3] = 0
    img_rgba = Image.fromarray(arr, 'RGBA')

    # ── assemble: map left, side strip right, bottom strip below ─────────────
    combined = Image.new('RGBA', (total_w, IMAGE_HEIGHT + strip_h), (0, 0, 0, 0))
    combined.paste(img_rgba,   (0, 0),              img_rgba)
    combined.paste(side_strip, (IMAGE_WIDTH, 0),    side_strip)
    combined.paste(strip,      (0, IMAGE_HEIGHT),   strip)
    return combined


# ── per-town processing ───────────────────────────────────────────────────────

def process_town(town_dir: Path, town_name: str, client, output_path: Path):
    traj_dir = town_dir / 'trajectories'
    if not traj_dir.exists():
        print(f'  No trajectories/ folder in {town_dir}, skipping.')
        return

    traj_files = sorted(traj_dir.glob('*.json'))
    if not traj_files:
        print(f'  No JSON files in {traj_dir}, skipping.')
        return

    if MODELS is not None:
        traj_files = [f for f in traj_files if f.stem in MODELS]
        # preserve the order defined in MODELS
        traj_files = sorted(traj_files, key=lambda f: MODELS.index(f.stem))

    print(f'  Loading {len(traj_files)} models for {town_name}...')
    models = {}
    model_stems = []
    for f in traj_files:
        label = MODEL_LABELS.get(f.stem, f.stem)
        with open(f) as fp:
            models[label] = json.load(fp)
        model_stems.append(f.stem)

    # Global bounding box across all models
    all_x = [p['x'] for pts in models.values() for p in pts]
    all_y = [p['y'] for pts in models.values() for p in pts]
    min_x, max_x = min(all_x) - PADDING, max(all_x) + PADDING
    min_y, max_y = min(all_y) - PADDING, max(all_y) + PADDING

    camera_x = (min_x + max_x) / 2
    camera_y = (min_y + max_y) / 2
    traj_radius = max(max_x - camera_x, max_y - camera_y)
    camera_z = CAMERA_HEIGHT_FACTOR * np.tan(np.radians(90 - CAMERA_FOV / 2)) * traj_radius
    mtp = IMAGE_WIDTH / (2 * math.tan(math.radians(CAMERA_FOV) / 2) * camera_z)

    dot_r      = int(_DOT_RADIUS       * DRAW_SCALE)
    arrow_len  = int(_ARROW_LENGTH     * DRAW_SCALE)
    arrow_head = int(_ARROW_HEAD       * DRAW_SCALE)
    start_r    = int(_START_DOT_RADIUS * DRAW_SCALE)
    ep_font_sz = int(_EPISODE_FONT_SZ  * DRAW_SCALE)

    serif_candidates = [
        '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    ]
    ep_font = ImageFont.load_default()
    for path in serif_candidates:
        try:
            ep_font = ImageFont.truetype(path, ep_font_sz)
            break
        except Exception:
            continue

    print(f'  Capturing background ({town_name})...')
    bg = capture_background(client, town_name, camera_x, camera_y, camera_z)
    image = Image.fromarray(bg)
    draw  = ImageDraw.Draw(image)

    model_names = list(models.keys())
    _fb_idx = 0
    colors = []
    for stem in model_stems:
        if stem in MODEL_COLORS:
            colors.append(MODEL_COLORS[stem])
        else:
            colors.append(_FALLBACK_PALETTE[_fb_idx % len(_FALLBACK_PALETTE)])
            _fb_idx += 1

    # ── draw trajectories (dots, subsampled) ──────────────────────────────────
    for color, (name, traj) in zip(colors, models.items()):
        for i, p in enumerate(traj):
            if p['is_infraction_terminal']:
                continue
            if i % TRAJ_STEP != 0:
                continue
            px, py = world_to_pixel(p['x'], p['y'], camera_x, camera_y, mtp)
            draw_dot(draw, px, py, dot_r, color)

    # ── draw episode start markers (black fill, white outline, per-episode shape) ──
    for name, traj in models.items():
        seen_episodes: set = set()
        for p in traj:
            ep = p['episode']
            if ep in seen_episodes:
                continue
            seen_episodes.add(ep)
            px, py = world_to_pixel(p['x'], p['y'], camera_x, camera_y, mtp)
            draw_marker(draw, px, py, start_r,
                        fill_color=(0, 0, 0), outline_color=(255, 255, 255),
                        episode_idx=ep)

    # ── draw non-infraction episode ends (white fill, black outline, per-episode shape) ──
    # Drawn AFTER start markers so they appear on top when positions overlap.
    for name, traj in models.items():
        by_episode: dict = defaultdict(list)
        for p in traj:
            by_episode[p['episode']].append(p)
        for ep_id, ep_steps in by_episode.items():
            last = ep_steps[-1]
            if last['is_infraction_terminal']:
                continue
            px, py = world_to_pixel(last['x'], last['y'], camera_x, camera_y, mtp)
            draw_marker(draw, px, py, start_r,
                        fill_color=(255, 255, 255), outline_color=(0, 0, 0),
                        episode_idx=ep_id)

    # ── draw infraction arrows on top ─────────────────────────────────────────
    for color, (name, traj) in zip(colors, models.items()):
        for p in traj:
            if not p['is_infraction_terminal']:
                continue
            px, py = world_to_pixel(p['x'], p['y'], camera_x, camera_y, mtp)
            draw_arrow(draw, px, py, p['yaw'], arrow_len, arrow_head, color)

    # ── legend ────────────────────────────────────────────────────────────────
    image = add_legend(image, model_names, colors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path.as_posix())
    n_inf = sum(sum(1 for p in traj if p['is_infraction_terminal']) for traj in models.values())
    print(f'  Saved: {output_path}  ({n_inf} infraction arrows across all models)')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot multi-model trajectories on CARLA map.')
    parser.add_argument('metrics_dir', nargs='?', default=str(DEFAULT_METRICS_DIR),
                        help='Folder with town01/, town02/ subfolders (default: eval_metrics_infraction)')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        sys.exit(f'ERROR: {metrics_dir} does not exist.')

    # Auto-discover town subfolders
    town_dirs = sorted(d for d in metrics_dir.iterdir()
                       if d.is_dir() and d.name.lower().startswith('town'))
    if not town_dirs:
        sys.exit(f'ERROR: no town* subdirectories found in {metrics_dir}.')

    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    out_base = OUTPUT_DIR / metrics_dir.name

    for town_dir in town_dirs:
        # town01 → Town01
        town_name = town_dir.name[0].upper() + town_dir.name[1:]
        out_path = out_base / f'{town_dir.name}.png'
        print(f'\nProcessing {town_name}...')
        process_town(town_dir, town_name, client, out_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
