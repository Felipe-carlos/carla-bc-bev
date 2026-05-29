# FiLM Conditioning com Comando de Navegação no CVT Encoder

## Problema

O CVT constrói o BEV via cross-view attention: features extraídas das câmeras servem como **keys e values**, e queries vêm do BEV prior. O mecanismo não sabe qual manobra o veículo deve executar — ele apenas integra evidência visual.

O comando de navegação (`cmd`) carrega essa intenção antecipadamente: LEFT, RIGHT, STRAIGHT, LANEFOLLOW. A ideia é condicioná-lo **antes** da atenção, para que o cross-view já saiba qual manobra procurar nas imagens ao construir o BEV.

---

## Por que FiLM

FiLM (Feature-wise Linear Modulation) modula os features canal a canal com scale e shift derivados do `cmd`:

```
x_out = (1 + γ(cmd)) ⊙ x  +  β(cmd)
```

- `γ` e `β` são produzidos por uma projeção linear a partir do embedding do `cmd`
- O termo `γ ⊙ x` é **multiplicativo** — se `γ_c > 0`, o canal `c` é amplificado; se `γ_c < 0`, suprimido. A magnitude do efeito depende do valor atual de `x`, tornando o conditioning **context-sensitive**
- `β` é um shift aditivo global por canal

### Por que não concatenação

Concatenar um vetor global às features espaciais e aplicar conv 1×1 produz:

```
output[h,w] = W_feat · feat[h,w]  +  W_cmd · cmd  +  bias
```

O termo `W_cmd · cmd` é **constante para todos os pixels** — é o mesmo offset independente do conteúdo de `feat[h,w]`. Matematicamente equivale ao `β` do FiLM sem o `γ`, com o custo adicional de replicar a informação global em 14×30 posições por câmera.

### Zero-init para estabilidade

Inicializar os pesos da projeção `(γ, β)` com zeros faz com que o modelo comece com `x_out = x`, ignorando o `cmd` inicialmente. O gradiente gradualmente aprende quando e quanto usá-lo, evitando instabilidade nos primeiros steps de treino.

---

## Onde injetar

O ponto de injeção é **após o backbone e antes da cross-view attention**. Isso garante que as features que servem de keys e values na atenção já estejam condicionadas pelo comando — o mecanismo de atenção câmera→BEV opera com informação de intenção de manobra embutida.

Injetar após a cross-view (no BEV resultante) seria menos expressivo: a atenção já teria integrado as features sem conhecer o comando.

---

## Implementação

```python
class CmdConditionedEncoder(Encoder):
    """
    Estende o Encoder base adicionando FiLM conditioning com
    o comando de navegação nas features do backbone, antes da
    cross-view attention.
    """

    def __init__(self, backbone, cross_view, bev_embedding,
                 dim=128, middle=[2], scale=1.0,
                 n_cmd_classes=6):
        super().__init__(backbone, cross_view, bev_embedding,
                         dim=dim, middle=middle, scale=scale)

        # Embedding aprendido por classe de comando
        self.cmd_emb = nn.Embedding(n_cmd_classes, dim)

        # Uma camada FiLM por escala de feature do backbone
        feat_dims = [
            self.down(torch.zeros(s)).shape[1]
            for s in self.backbone.output_shapes
        ]
        self.cmd_film = nn.ModuleList([
            nn.Linear(dim, 2 * fd)
            for fd in feat_dims
        ])

        # Zero-init: começa como identidade
        for layer in self.cmd_film:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)
        I_inv = batch['intrinsics'].inverse()
        E_inv = batch['extrinsics'].inverse()

        # 1. Extrair features do backbone
        features = [self.down(y) for y in self.backbone(self.norm(image))]
        # features[i]: (B·n, feat_dim_i, h_i, w_i)

        # 2. FiLM: modular cada escala com γ e β derivados do cmd
        if 'cmd' in batch:
            cmd_idx  = batch['cmd'].argmax(dim=-1)     # (B,)
            cmd_feat = self.cmd_emb(cmd_idx)            # (B, dim)

            modulated = []
            for feat, film_layer in zip(features, self.cmd_film):
                params      = film_layer(cmd_feat)             # (B, 2·feat_dim)
                gamma, beta = params.chunk(2, dim=-1)          # (B, feat_dim) cada

                # Expandir para (B·n, feat_dim, 1, 1)
                gamma = gamma[:, None].expand(-1, n, -1).reshape(b*n, -1, 1, 1)
                beta  =  beta[:, None].expand(-1, n, -1).reshape(b*n, -1, 1, 1)

                modulated.append((1.0 + gamma) * feat + beta)
                # shape preservado: (B·n, feat_dim_i, h_i, w_i)

            features = modulated

        # 3. BEV prior
        x = self.bev_embedding.get_prior()
        x = repeat(x, '... -> b ...', b=b)

        # 4. Cross-view attention com features já condicionadas
        for cross_view, feat, layer in zip(self.cross_views, features, self.layers):
            feat = rearrange(feat, '(b n) ... -> b n ...', b=b, n=n)
            x    = cross_view(x, self.bev_embedding, feat, I_inv, E_inv)
            x    = layer(x)

        return x
```

---

## Fluxo de dados

```
cmd (B, 6) one-hot
    │
    ▼ argmax
cmd_idx (B,)
    │
    ▼ Embedding
cmd_feat (B, dim)
    │
    ▼ Linear → chunk
γ (B, feat_dim)    β (B, feat_dim)
    │
    ▼ expand(n câmeras) + reshape + broadcast espacial
γ (B·n, feat_dim, 1, 1)
    │
    ▼ FiLM
features_moduladas = (1 + γ) ⊙ features + β
    │                ← shape inalterado: (B·n, feat_dim, h, w)
    ▼
Cross-view attention — keys e values condicionados pelo cmd
    │
    ▼
BEV (B, dim, H, W)
```

O shape das features nunca muda. A `CrossViewAttention` não requer nenhuma alteração — recebe features com as mesmas dimensões, mas cujos canais foram reorganizados de acordo com a intenção de manobra.

---

## O que este encoder precisa receber em `batch`

| Chave | Shape | Tipo | Origem |
|---|---|---|---|
| `image` | `(B, n, C, H, W)` | float32 | câmeras RGB |
| `intrinsics` | `(B, n, 3, 3)` | float32 | matrizes de câmera |
| `extrinsics` | `(B, n, 4, 4)` | float32 | matrizes de câmera |
| `cmd` | `(B, 6)` | float32 one-hot | comando de navegação |

`cmd` é **opcional** — se ausente do batch, o bloco FiLM é ignorado e o encoder se comporta identicamente ao base.
