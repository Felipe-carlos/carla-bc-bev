# Migração — VanillaTraj: Injeção Direta de Trajetória no BEV Embedding

## Motivação

Na arquitetura vanilla, a trajetória é renderizada como pontos brancos em canvas preto e empilhada como uma 5ª câmera RGB. Isso tem três problemas:

1. **EfficientNet-B4 é pré-treinado no ImageNet** — espera imagens naturais. Pontos brancos em fundo preto são entrada degenerada.
2. **O mapeamento geométrico é um hack** — a trajetória recebe intrinsics/extrinsics de uma câmera BEV virtual, mas ela não é uma imagem perspectiva.
3. **O sinal se perde** — 4 câmeras RGB ricas contra 1 imagem esparsa de trajetória.

A solução: remover a trajetória do stack de imagens e injetá-la diretamente no BEV embedding prior via `WaypointMLP` + Gaussianas 2D.

---

## Arquivos modificados

### 1. `/home/felipe_cds/cvt-6ch/model/encoder.py`

**O que foi adicionado** — três novos símbolos públicos antes da classe `Encoder` (linha ~295):

#### `class WaypointMLP(nn.Module)`
MLP que mapeia cada waypoint `(x, y)` em metros para um vetor de features de dimensão `dim`.

```python
class WaypointMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, wp: torch.Tensor) -> torch.Tensor:
        return self.net(wp)
```

#### `def build_bev_grid(...) -> torch.Tensor`
Constrói um grid `(2, h, w)` de coordenadas ego-frame em metros, espelhando o layout interno de `BEVEmbedding.grid`. Deve ser registrado como buffer não-persistente no módulo que o usa.

```python
def build_bev_grid(
    h: int, w: int,
    bev_h: int, bev_w: int,
    h_meters: float, w_meters: float,
    offset: float = 0.0,
) -> torch.Tensor:
```

#### `def traj_to_bev_signal(...) -> torch.Tensor`
Converte waypoints normalizados `(÷100 m)` em um sinal `(B, dim, H, W)` pronto para ser somado ao BEV prior. Para cada waypoint: calcula features via MLP, cria uma Gaussiana 2D centrada no waypoint sobre o grid BEV e acumula `feat_i * gaussian_i`.

```python
def traj_to_bev_signal(
    traj_norm: torch.Tensor,   # (B, n_waypoints*2)
    mlp: WaypointMLP,
    bev_grid: torch.Tensor,    # (2, H, W) — buffer pré-construído
    n_waypoints: int,
    sigma: float,
    dim: int,
) -> torch.Tensor:             # (B, dim, H, W)
```

> **Nota sobre sigma:** a resolução do grid prior é 256 px / 100 m = **3.125 m/célula**. Com `sigma = 6.0 m ≈ 2 células/σ`, cada waypoint ativa uma vizinhança de ~11×11 células — localizado o suficiente sem ser agressivo. Ajuste para baixo se o sinal ficar difuso demais.

#### `class TrajEncoder(Encoder)`
Subclasse de `Encoder` que adiciona a injeção de trajetória. Aceita `n_waypoints` e `sigma` como parâmetros de configuração.

```python
class TrajEncoder(Encoder):
    def __init__(
        self,
        backbone,
        cross_view: dict,
        bev_embedding: dict,
        dim: int = 128,
        middle: List[int] = [2],
        scale: float = 1.0,
        n_waypoints: int = 5,   # ← configurável na model definition
        sigma: float = 6.0,     # ← configurável na model definition
    ):
```

No `__init__`, além de chamar `super().__init__()`, instancia `self.waypoint_mlp` e registra `self.bev_grid` como buffer não-persistente com shape `(2, h_prior, w_prior)`.

No `forward`, o sinal é injetado **depois** do `repeat` do prior e **antes** do loop de cross-view attention:

```python
x = self.bev_embedding.get_prior()
x = repeat(x, '... -> b ...', b=b)

if 'traj' in batch:          # graceful degradation se a chave estiver ausente
    traj_signal = traj_to_bev_signal(...)
    x = x + traj_signal

for cross_view, feature, layer in zip(...):
    ...
```

**O que foi alterado no `Encoder` existente:** apenas remoção dos `print` de debug que existiam no `forward`. A assinatura e o comportamento são idênticos ao original.

---

### 2. `/home/felipe_cds/cvt-6ch/model/__init__.py`

**Linha 1** — adicionado `TrajEncoder` ao import:
```python
# antes
from model.encoder import Encoder
# depois
from model.encoder import Encoder, TrajEncoder
```

**Novo bloco** — `class ModelBuilderTraj()` inserido entre `ModelBuilder` e `ModelBuilderLarger`. Funciona igual ao `ModelBuilder` mas instancia um `TrajEncoder` e repassa `n_waypoints` e `sigma`:

```python
class ModelBuilderTraj():
    def __init__(
        self,
        masks=False, reduction=4, backbone=None, low_stride=False,
        decoder=None, dim_output=6,
        n_waypoints=5, sigma=6.0,
    ):
        config = Config(masks=masks, reduction=reduction, low_stride=low_stride)
        backbone = backbone if backbone is not None else config.backbone
        encoder = TrajEncoder(
            backbone=backbone,
            cross_view=config.cross_view,
            bev_embedding=config.bev_embedding,
            dim=config.encoder_dim,
            middle=[2],
            scale=1.0,
            n_waypoints=n_waypoints,
            sigma=sigma,
        )
        if decoder is None:
            decoder = Decoder(
                dim=config.encoder_dim,
                blocks=[128, 128, 64],
                residual=True,
                factor=2,
            )
        self.network = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=dim_output,
            dim_last=64,
        )
    def get_net(self):
        return self.network
```

---

### 3. `/home/felipe_cds/cvt-6ch/model/model_definitions/vanilla_traj.py` *(arquivo novo)*

Model definition do experimento. `n_waypoints` e `sigma` são atributos de classe — altere-os por herança ou edição direta para testar variantes sem tocar na arquitetura.

```python
from .base import BaseModelDefinition
from model.cvt import CrossViewTransformer
from model import ModelBuilderTraj
import torch.nn as nn


class VanillaTraj(BaseModelDefinition):
    n_waypoints: int = 5
    sigma: float = 6.0

    @property
    def name(self) -> str:
        return "vanilla_traj"

    def get_loss(self):
        return nn.BCEWithLogitsLoss(reduction='none')

    def get_model(self) -> CrossViewTransformer:
        builder = ModelBuilderTraj(n_waypoints=self.n_waypoints, sigma=self.sigma)
        return builder.get_net()
```

Checkpoints salvos em `ckpt-vanilla_traj/`, logs em `logs/vanilla_traj.json`.

---

### 4. `/home/felipe_cds/cvt-6ch/model/model_definitions/__init__.py`

Adicionadas duas linhas:
```python
from .vanilla_traj import VanillaTraj   # import
"VanillaTraj"                           # entrada no __all__
```

---

### 5. `/home/felipe_cds/cvt-6ch/dataset_def/expert_dataset.py`

#### `get_intrinsics` e `get_extrinsics`

Adicionado parâmetro `traj_injection=False`. Quando `True`, a câmera BEV virtual (`'bev'`) é removida do `stack_order`, retornando 4 matrizes ao invés de 5.

```python
# antes
def get_intrinsics(obs_configs, bev_resize):
    stack_order = ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb', 'bev']

# depois
def get_intrinsics(obs_configs, bev_resize, traj_injection=False):
    stack_order = ['left_rgb', 'central_rgb', 'right_rgb', 'rear_rgb']
    if not traj_injection:
        stack_order = stack_order + ['bev']
```

Mesma lógica em `get_extrinsics`.

#### `ExpertDataset.__init__`

Adicionado parâmetro `use_traj_injection=False`, armazenado como `self.use_traj_injection`.

#### `ExpertDataset.__getitem__`

Comportamento controlado pelo flag:

| `use_traj_injection` | `image` shape | `extrinsics` shape | `intrinsics` shape | chave `traj` |
|---|---|---|---|---|
| `False` (padrão) | `(5, 3, H, W)` | `(5, 4, 4)` | `(5, 3, 3)` | ausente |
| `True` | `(4, 3, H, W)` | `(4, 4, 4)` | `(4, 3, 3)` | `(n_waypoints*2,)` |

Quando `True`: a trajetória **não** é renderizada como imagem e o tensor raw normalizado é adicionado ao batch:
```python
if self.use_traj_injection:
    images = th.stack([left_rgb, central_rgb, right_rgb, rear_rgb])
else:
    traj_plot_rgb = traj_plotter_rgb(...) / 255.0
    images = th.stack([left_rgb, central_rgb, right_rgb, rear_rgb, traj_plot_rgb])

# ...

if self.use_traj_injection:
    obs_dict['traj'] = th.tensor(state_dict['traj'], dtype=th.float32)
```

O campo `state_dict['traj']` vem do `episode.json` e contém os waypoints normalizados por `÷100 m` (ex: 5 waypoints × (x, y) → shape `(10,)`).

---

### 6. `/home/felipe_cds/cvt-6ch/dataset_def/__init__.py`

Adicionado `use_traj_injection=False` ao `ConfigDatasets.__init__`, repassado para todos os três splits:

```python
# antes
class ConfigDatasets:
    def __init__(self, datasest_folder: Path, multi_label=False, batch_size=16, use_kde=False):

# depois
class ConfigDatasets:
    def __init__(self, datasest_folder: Path, multi_label=False, batch_size=16, use_kde=False, use_traj_injection=False):
```

---

## Como usar no script de treino

```python
from model.model_definitions.vanilla_traj import VanillaTraj
from dataset_def import ConfigDatasets
from pathlib import Path

model_config = VanillaTraj()

data = ConfigDatasets(
    datasest_folder=Path('/home/jovyan/privado/cvt-6ch/datasets'),
    batch_size=64,
    use_traj_injection=True,   # ← ativa a nova modalidade
)

network = model_config.get_model().cuda()
criterion = model_config.get_loss().cuda()
```

---

## Resumo das shapes de batch

| Chave | Vanilla (original) | VanillaTraj (novo) |
|---|---|---|
| `image` | `(B, 5, 3, 224, 480)` | `(B, 4, 3, 224, 480)` |
| `extrinsics` | `(B, 5, 4, 4)` | `(B, 4, 4, 4)` |
| `intrinsics` | `(B, 5, 3, 3)` | `(B, 4, 3, 3)` |
| `bev` | `(B, 6, 256, 256)` | `(B, 6, 256, 256)` |
| `kde_w` | `(B,)` | `(B,)` |
| `traj` | ausente | `(B, 10)` |
| **saída do modelo** | `(B, 6, 256, 256)` | `(B, 6, 256, 256)` |

Decoder, loss e código de avaliação não precisam de alteração.
