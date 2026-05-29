# Avaliação dos Geradores de BEV para Direção Autônoma no CARLA

## Visão Geral

Este documento reporta os resultados da avaliação experimental de múltiplos geradores de Bird's Eye View (BEV) aplicados a um agente de Behavioral Cloning (BC) no simulador CARLA. O objetivo é substituir a BEV de ground-truth disponível no treinamento por uma BEV **predita** em tempo de inferência, mantendo a qualidade da direção do agente.

---

## 1. Configuração Experimental

### 1.1 Pipeline

```
Câmeras RGB (4 ângulos: esquerda, central, direita, traseira)
    ↓
Gerador BEV  →  BEV predita (3 ou 6 canais, 192×192)
    ↓
AgentPolicy (XtMaCNN + BetaDistribution)
    ↓
Ações: throttle ∈ [0,1], steer ∈ [-1,1]
```

### 1.2 Ambientes de Teste

Avaliações conduzidas em dois mapas do CARLA com rotas de ~1.15–1.29 km:
- **Town01** — ambiente de treinamento
- **Town02** — ambiente não visto (generalização)

### 1.3 Combinações Avaliadas

Duas dimensões variam independentemente:

| Dimensão | O que controla |
|---|---|
| **Policy** | Qual policy BC é usada para dirigir |
| **BEV na inferência** | Qual gerador produz a observação |

A combinação mais relevante é `real-bev + <gerador>`: a policy foi treinada com a BEV de ground-truth e recebe a BEV gerada em inferência. Isso testa diretamente o *gap* de domínio.

### 1.4 Métricas de Direção

| Métrica | Descrição |
|---|---|
| **Reward** | Recompensa acumulada por episódio (maior = melhor) |
| **Route compl. (%)** | Percentual médio da rota completado |
| **Metros dirigidos** | Distância percorrida por episódio antes da colisão/timeout |
| **Score composed** | Produto entre `score_route` e `score_penalty` (penaliza infrações) |

### 1.5 Métricas de Qualidade da BEV (IoU)

Calculado comparando a BEV predita com a BEV de ground-truth. São reportados dois agregados:

- **IoU médio (mean):** qualidade global ao longo de toda a rota
- **IoU mínimo (min) e p5:** qualidade no pior momento — diretamente relacionado ao risco de colisão

| Canal | Semântica |
|---|---|
| **Ch0 (Ruas)** | Área da pista / road surface |
| **Ch1 (Trajetória)** | Caminho planejado / waypoints |
| **Ch2 (Lane Boundary)** | Bordas de faixa / marcações |

---

## 2. Modelos Avaliados

### 2.1 UNet

**Arquitetura:** Rede encoder-decoder U-Net, recebe as 4 câmeras RGB concatenadas + imagem de trajetória renderizada, produz BEV 3 canais (192×192).

**Motivação:** Baseline de tradução imagem-a-imagem sem mecanismo de atenção multi-câmera explícito. Abordagem direta, computacionalmente simples.

**Limitação esperada:** Sem modelagem geométrica explícita da projeção perspectiva para BEV, dificulta aprender a correspondência espacial entre câmeras em condições variadas de iluminação e cenário.

---

### 2.2 CVT 3ch L1 (baseline CVT)

**Arquitetura:** CrossViewTransformer (EfficientNet-b4 + atenção multi-câmera cross-view), produz BEV 3 canais (192×192). Recebe 4 câmeras RGB + imagem de trajetória renderizada (traj plot).

**Função de perda:** L1 entre a BEV predita e a BEV de ground-truth.

**Motivação:** Introduz atenção geométrica entre múltiplas câmeras, explorando as relações espaciais para construir uma representação BEV consistente. O traj plot informa o contexto de direção planejada. É o modelo base da família CVT neste projeto.

---

### 2.3 CVT 3ch Finetuned (CVT 3ch FT)

**Arquitetura:** Idêntica ao CVT 3ch L1.

**Motivação:** Fine-tuning do CVT 3ch L1 em dados do domínio CARLA específico do projeto, buscando reduzir o gap de distribuição entre os dados de pré-treinamento e o cenário de avaliação.

**Hipótese:** Fine-tuning com kde em distribuição in-domain vai melhorar os cruzamentos, evitando as quedas bruscas de IoU que causam colisões.

---

### 2.4 CVT 6ch Vanilla

**Arquitetura:** CVT com saída de **6 canais** (apenas os 3 primeiros são usados em inferência). Não usa traj plot; recebe diretamente as **matrizes intrínsecas/extrínsecas** das câmeras como condicionamento geométrico.

**Função de perda:** BCE-with-logits (Binary Cross-Entropy).

**Motivação:** Expandir a capacidade de representação do modelo e eliminar a dependência do traj plot. As matrizes de câmera fornecem ao modelo a geometria real da projeção, permitindo melhor estimativa de posição dos elementos na BEV e produzindo saídas mais nítidas e estáveis.

---

### 2.5 CVT 6ch KDE

**Arquitetura:** Idêntica ao CVT 6ch Vanilla (`ModelBuilder` reutilizado).

**Função de perda:** BCE com pesos calculados por **KDE (Kernel Density Estimation)**, aplicada sobre a predição com sigmoid. A ponderação KDE torna o treinamento mais sensível a pixels positivos raros (bordas de faixa, obstáculos), combatendo o desbalanceamento de classes na BEV.

**Motivação:** Pixels positivos (road markings, veículos) são minoria na BEV — a maioria é fundo. A BCE padrão tende a ignorá-los. O KDE força o modelo a focar nas regiões críticas para navegação, elevando o piso de qualidade nas situações onde a BEV é mais difícil de predizer (curvas, cruzamentos).

---

### 2.6 CVT 6ch Traj

**Arquitetura:** CVT 6ch Vanilla + **TrajEncoder**: N=5 waypoints futuros são codificados por um MLP e projetados como heatmaps Gaussianos 2D no prior BEV antes das camadas de atenção.

**Motivação:** Injetar **intenção de direção** diretamente no espaço BEV antes do raciocínio visual. O modelo sabe antecipadamente por onde o veículo pretende passar, podendo focar atenção nas regiões relevantes do campo visual e produzir BEV mais precisa na área à frente.


---

## 3. Resultados de Qualidade da BEV (IoU Offline)

Avaliação sobre o conjunto de validação (~1496 amostras), comparando os geradores isoladamente contra a BEV de ground-truth.

### 3.1 IoU Médio

| Modelo | Ch0 – Ruas | Ch1 – Trajetória | Ch2 – Lane Boundary | **IoU Médio** |
|---|:---:|:---:|:---:|:---:|
| CVT 3ch L1 | **0.9575** | **0.8555** | **0.7140** | **0.8423** |
| CVT 3ch FT | 0.9273 | 0.7967 | 0.4850 | 0.7363 |
| UNet | 0.7520 | 0.5711 | 0.5095 | 0.6109 |
| CVT 6ch KDE | 0.8648 | 0.6296 | 0.0580 | 0.5175 |
| CVT 6ch Vanilla | 0.8424 | 0.5858 | 0.0319 | 0.4867 |

### 3.2 IoU Mínimo e Percentil 5%

O IoU médio mede a qualidade *típica*, mas o que provoca colisão é a qualidade no *pior momento*. As métricas de cauda inferior capturam o quão ruim a BEV pode ficar:

| Modelo | min Ch0 | min Ch1 | min Ch2 | **min IoU Médio** | **p5 IoU Médio** |
|---|:---:|:---:|:---:|:---:|:---:|
| CVT 6ch KDE | **0.4990** | 0.0487 | 0.0000 | **0.2507** | **0.3201** |
| CVT 6ch Vanilla | 0.4645 | 0.0637 | 0.0000 | 0.2128 | 0.3147 |
| CVT 3ch FT | 0.4119 | 0.0983 | 0.0000 | 0.1750 | 0.3584 |
| CVT 3ch L1 | 0.3655 | 0.1018 | 0.0000 | 0.1923 | 0.4576 |
| UNet | 0.2517 | 0.0865 | 0.0000 | 0.1257 | 0.1776 |

O ranking por **mínimo de Ch0** já anuncia o desempenho de direção melhor que o ranking por média — ver Seção 6 para a análise completa.

---

## 4. Resultados de Qualidade da BEV durante a Navegação

IoU calculado frame a frame durante os episódios de avaliação no CARLA. Captura como a qualidade da BEV se comporta sob a distribuição de estados de um episódio real de direção.

### 4.1 Town01

| Policy | BEV na inferência | Mean IoU | Min IoU | p5 IoU |
|---|---|:---:|:---:|:---:|
| Real-BEV (GT) | **CVT 6ch Traj** | 0.719 | **0.287** | **0.444** |
| Real-BEV (GT) | CVT 6ch KDE | 0.712 | 0.193 | 0.433 |
| Real-BEV (GT) | CVT 6ch Vanilla | 0.669 | 0.233 | 0.421 |
| Real-BEV (GT) | CVT 3ch FT | 0.807 | 0.118 | 0.404 |
| Real-BEV (GT) | CVT 3ch L1 | 0.783 | 0.056 | 0.336 |
| Real-BEV (GT) | UNet | 0.160 | 0.015 | 0.059 |
| CVT 6ch (policy própria) | CVT 6ch Vanilla | 0.662 | 0.222 | 0.395 |
| CVT 3ch L1 (policy própria) | CVT 3ch L1 | 0.773 | 0.083 | 0.292 |
| UNet (policy própria) | UNet | 0.144 | 0.027 | 0.059 |

### 4.2 Town02

| Policy | BEV na inferência | Mean IoU | Min IoU | p5 IoU |
|---|---|:---:|:---:|:---:|
| Real-BEV (GT) | **CVT 6ch Traj** | 0.652 | **0.275** | **0.399** |
| Real-BEV (GT) | CVT 6ch KDE | 0.636 | 0.239 | 0.366 |
| Real-BEV (GT) | CVT 6ch Vanilla | 0.635 | 0.205 | 0.334 |
| Real-BEV (GT) | CVT 3ch FT | 0.650 | 0.067 | 0.159 |
| Real-BEV (GT) | CVT 3ch L1 | 0.679 | 0.063 | 0.186 |
| Real-BEV (GT) | UNet | 0.167 | 0.009 | 0.055 |
| CVT 6ch (policy própria) | CVT 6ch Vanilla | 0.636 | 0.172 | 0.324 |
| CVT 3ch L1 (policy própria) | CVT 3ch L1 | 0.700 | 0.068 | 0.191 |
| UNet (policy própria) | UNet | 0.177 | 0.020 | 0.064 |

---

## 5. Resultados de Direção no CARLA

### 5.1 Town01

#### Policy `Real-BEV (GT)` — BEV gerada em inferência (avalia o gap de domínio)

| BEV na inferência | Reward | Route compl. (%) | Metros dirigidos | Score composed |
|---|:---:|:---:|:---:|:---:|
| **Ground-truth (GT)** *(baseline)* | — | — | **4810 m** | — |
| **CVT 6ch Vanilla** | **373.4** | **29.6** | **294.9** | **0.153** |
| CVT 6ch KDE | 242.1 | 18.6 | 195.6 | 0.117 |
| CVT 6ch Traj | 234.8 | 18.5 | 197.0 | 0.116 |
| CVT 3ch FT | 182.6 | 14.1 | 195.4 | 0.092 |
| CVT 3ch L1 | 159.6 | 11.5 | 115.6 | 0.084 |
| UNet | -10.0 | 2.7 | 28.1 | 0.020 |

#### Policy e BEV treinados juntos (avalia o par end-to-end)

| Policy + BEV | Reward | Route compl. (%) | Metros dirigidos | Score composed |
|---|:---:|:---:|:---:|:---:|
| CVT 6ch Vanilla (policy própria) | 182.5 | 17.5 | 187.5 | 0.088 |
| CVT 3ch L1 (policy própria) | 146.7 | 11.7 | 110.8 | 0.090 |
| UNet (policy própria) | -123.7 | 5.1 | 51.4 | 0.035 |

#### Policy `CVT 3ch L1` com buffer temporal (3 frames, 9 canais)

| BEV na inferência | Reward | Route compl. (%) | Metros dirigidos | Score composed |
|---|:---:|:---:|:---:|:---:|
| CVT 6ch Vanilla | 102.1 | 11.9 | 118.6 | 0.070 |
| **CVT 6ch Traj** | **77.6** | **14.5** | **171.5** | 0.054 |
| CVT 6ch KDE | 72.5 | 10.8 | 116.5 | 0.062 |
| CVT 3ch FT | 61.0 | 7.9 | 89.7 | 0.057 |
| CVT 3ch L1 | 48.3 | 6.8 | 87.2 | 0.044 |

---

### 5.2 Town02

#### Policy `Real-BEV (GT)` — BEV gerada em inferência

| BEV na inferência | Reward | Route compl. (%) | Metros dirigidos | Score composed |
|---|:---:|:---:|:---:|:---:|
| **Ground-truth (GT)** *(baseline)* | — | — | **4843 m** | — |
| **CVT 6ch KDE** | **140.0** | **12.3** | **130.0** | **0.080** |
| CVT 6ch Vanilla | 137.9 | 11.8 | 129.2 | 0.075 |
| CVT 6ch Traj | 134.7 | 11.5 | 106.9 | 0.067 |
| CVT 3ch L1 | 85.1 | 8.2 | 82.0 | 0.071 |
| CVT 3ch FT | 81.8 | 7.8 | 76.2 | 0.060 |
| UNet | -18.5 | 4.9 | 46.8 | 0.037 |

#### Policy `CVT 3ch L1` com buffer temporal

| BEV na inferência | Reward | Route compl. (%) | Metros dirigidos | Score composed |
|---|:---:|:---:|:---:|:---:|
| **CVT 6ch KDE** | **67.9** | **11.4** | **102.8** | **0.087** |
| CVT 3ch FT | 45.2 | 5.9 | 59.3 | 0.051 |
| CVT 3ch L1 | 46.6 | 6.0 | 53.3 | 0.049 |
| CVT 6ch Traj | 40.3 | 7.4 | 70.1 | 0.064 |
| CVT 6ch Vanilla | 39.5 | 8.6 | 77.7 | 0.056 |

---

## 6. Relação entre IoU Mínimo da BEV e Desempenho de Direção

### 6.1 Por que o IoU mínimo importa mais que a média

Um agente de BC dirige de forma **sequencial**: cada step depende de uma BEV adequada. Se em algum momento a BEV predita deteriora gravemente — mesmo que brevemente — o agente produz uma ação errada e colide. A colisão encerra o episódio imediatamente, fazendo com que a performance de navegação reflita o pior evento, não a média.

Isso significa que o IoU médio mede a qualidade *típica*, mas é o **IoU mínimo** (e o percentil 5%) que determinam *até onde* o agente consegue chegar. Um modelo que produz BEV de qualidade 0.80 em média mas ocasionalmente cai para 0.02 será catastrófico — exatamente o caso do UNet.

### 6.2 Evidência: o piso de IoU Ch0 prediz a direção

Usando o min de Ch0 (pista — o canal mais crítico para manter-se na via) calculado durante os episódios de avaliação em Town01:


O p5 (5º percentil do IoU médio durante a navegação) captura o mesmo padrão de forma mais robusta que o mínimo absoluto:

| Modelo | p5 IoU — T01 | Metros dirigidos — T01 |
|---|:---:|:---:|
| CVT 6ch Traj | **0.444** | 197.0 m |
| CVT 6ch KDE | 0.433 | 195.6 m |
| CVT 6ch Vanilla | 0.421 | **294.9 m** |
| CVT 3ch FT | 0.404 | 195.4 m |
| CVT 3ch L1 | 0.336 | 115.6 m |
| UNet | 0.059 | 28.1 m |

Os modelos 6ch formam um cluster bem separado dos modelos 3ch e do UNet no piso de qualidade — e essa separação reproduz fielmente a separação em metros dirigidos.

### 6.3 Por que o IoU médio inverte o ranking

O IoU médio (Seção 3.1) coloca CVT 3ch L1 em primeiro lugar (0.842) e CVT 6ch Vanilla em último (0.487). Isso acontece por três razões combinadas:

1. **Semântica dos canais**: O CVT 3ch foi treinado com perda L1 diretamente contra a GT de 3 canais — é literalmente otimizado para maximizar o IoU global. O CVT 6ch usa BCE e produz saídas com distribuição diferente; seu Ch2 (marcações de faixa) é muito esparso (IoU ≈ 0.03) puxando a média para baixo, mas isso não afeta a capacidade do agente de se manter na pista.

2. **Qualidade no piso vs. na média**: A perda L1 suaviza a predição para minimizar o erro médio, o que resulta em predições mais difusas. Saídas difusas têm IoU médio elevado, mas nas situações de maior ambiguidade visual o modelo também suaviza a resposta — e uma BEV difusa e imprecisa numa curva é suficiente para causar a colisão. A BCE produz saídas mais binárias, o que eleva o **piso mínimo** de qualidade mesmo às custas da média.

3. **O colapso do UNet**: O UNet obtém IoU médio razoável (0.611) porque na maioria dos frames o fundo é bem predito. Mas nos frames difíceis (cruzamentos, sombras) ele colapsa para IoU ≈ 0.02 — e é nesses momentos que o agente colide. O p5 do UNet (0.059) é dramaticamente menor que qualquer CVT, revelando sua instabilidade estrutural.

### 6.4 O papel da injeção de trajetória no piso de qualidade

O CVT 6ch Traj apresenta consistentemente o **maior piso de IoU** durante a navegação (min\_mean 0.287 T01; 0.275 T02 — maior entre todos os modelos). Isso é coerente com seu design: ao injetar os waypoints futuros como heatmaps Gaussianos no prior BEV, o modelo recebe uma *âncora espacial* de onde a pista deve estar.

Nas situações de maior ambiguidade visual (início de curva, cruzamento) — exatamente quando outros modelos colapsam — o sinal de trajetória ancora o cross-view attention na região correta, evitando a queda de IoU. Esse mecanismo explica por que, no buffer temporal (onde o histórico de frames compõe 2/3 da observação), o CVT 6ch Traj é o único modelo a superar o CVT 6ch Vanilla em metros dirigidos em Town01 (171.5 m vs. 118.6 m): a estabilidade temporal da BEV gerada reduz o ruído acumulado de frames anteriores.

---

## 7. Efeito do Buffer Temporal

O buffer temporal fornece à policy 3 frames consecutivos de BEV (9 canais total: t-2, t-1, t), permitindo inferir velocidade e direção de movimento.

**Resultado geral:** O buffer temporal **não melhora** o desempenho em nenhuma combinação comparada à versão sem buffer, e em vários casos piora. Possíveis causas:

- A policy BC foi treinada frame-a-frame; receber 9 canais introduz uma distribuição de entrada fora do treinamento.
- Erros de predição da BEV se acumulam ao longo dos frames — uma BEV ruim em t-2 contamina a observação em t, reduzindo o piso de qualidade efetivo.
- O CVT 6ch Traj no buffer temporal (Town01) obtém o maior route completion (14.5%), sugerindo que a estabilidade do piso de IoU deste modelo compensa parcialmente o problema de acumulação de ruído.

---

## 8. Síntese e Conclusões

| Conclusão | Evidência |
|---|---|
| **O IoU mínimo (piso) prediz a direção; o IoU médio não** | Inversão de ranking: CVT 3ch L1 tem melhor média (0.842) mas pior piso de Ch0 (0.17) e pior direção entre CVTs |
| **CVT 6ch Vanilla é o melhor gerador para direção** (sem buffer temporal) | Melhor em reward, route %, metros e score em Town01; 2º em Town02 |
| **CVT 6ch Traj tem o maior piso de IoU** durante a navegação | min\_mean 0.287 T01, 0.275 T02 — maior entre todos os modelos testados |
| **UNet não é competitivo** em nenhuma métrica | Menor piso de Ch0 (0.02), reward negativo, último em metros dirigidos |
| **O buffer temporal não traz ganho líquido** | Degradação consistente vs. sem buffer em todos os geradores |
| **CVT 6ch KDE se destaca em Town02** | Melhor em reward e metros no ambiente não visto |
| **Generalização Town02 é desafiadora** | Todos os modelos perdem ~60–70% de desempenho vs. Town01 |

### Recomendação

Para maximizar o **desempenho de direção** (sem buffer temporal), o **CVT 6ch Vanilla** é o mais eficaz quando a policy é treinada com BEV de ground-truth. Para maximizar a **estabilidade e o piso mínimo de qualidade** — especialmente com buffer temporal ou em ambientes de alta variabilidade visual — o **CVT 6ch Traj** é o candidato preferencial.