# Trabalho final — Aprendizado Profundo

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)
![segmentation--models--pytorch](https://img.shields.io/badge/segmentation--models--pytorch-0.5.0-8A2BE2)
![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?logo=mlflow&logoColor=white)
![uv](https://img.shields.io/badge/package_manager-uv-DE5FE9?logo=astral&logoColor=white)

- **Instituição:** Universidade do Vale do Itajaí (UNIVALI)
- **Programa:** PPGCA — Programa de Pós-Graduação em Computação Aplicada
- **Disciplina:** Aprendizado Profundo
- **Professor:** Felipe Viel
- **Aluno:** Douglas Fabiã Martins

Este repositório contém a implementação do trabalho final da disciplina, dedicada à
segmentação semântica de plumas de metano em imagens hiperespectrais. O projeto
usa dados do STARCOP e compara arquiteturas próprias e pré-treinadas, mantendo o
ciclo de preparação dos dados, treinamento, avaliação, visualização e
acompanhamento dos experimentos.

## O que tem no projeto

### Dados e preparação

- carregamento de patches com quatro canais de entrada: `mag1c`,
  `TOA_AVIRIS_640nm`, `TOA_AVIRIS_550nm` e `TOA_AVIRIS_460nm`;
- normalização dos canais e preservação dos rótulos binários `labelbinary`;
- suporte aos datasets `starcop_mini` e `starcop_raw`;
- patches de `128 × 128` pixels, com sobreposição de `64` pixels;
- cache em memória e cache pré-computado em disco;
- geração de manifestos para o subconjunto R2;
- verificações de qualidade dos dados e dos manifestos de cenas.

Os datasets não são versionados neste repositório por causa do seu tamanho. O
procedimento de download e a estrutura esperada estão descritos na seção
[Preparação dos dados](#preparação-dos-dados).

### Modelos

As três arquiteturas produzem mapas de logits de um canal a partir de entradas
com quatro canais:

1. **E1 — TinyUNet:** U-Net pequena construída do zero, usada como baseline;
2. **E2 — U-Net + MobileNetV2:** encoder pré-treinado na ImageNet;
3. **E3 — LinkNet + MobileNetV3 Small Minimal:** encoder pré-treinado, seguindo a
   arquitetura comparada no trabalho de referência.

### Avaliação e visualização

O projeto inclui:

- métricas de pixel e de patch, incluindo precisão, recall, F1, matriz de
  confusão e AUPRC;
- avaliação por patch e por cena completa;
- calibração e seleção de limiar usando curvas precisão-recall;
- protocolo de avaliação por grupos de cenas (`strong`, `weak` e `plume_free`);
- figuras de arquitetura, distribuição de classes, curvas de treinamento,
  curvas precisão-recall, efeito do limiar e predições qualitativas;
- medição opcional de energia e potência durante a inferência.

## Como executar

### Localmente

Pré-requisitos: Python 3.12, [`uv`](https://docs.astral.sh/uv/) e espaço em disco
para os dados do STARCOP.

```bash
# instala as dependências e cria o ambiente virtual
uv sync
```

### Preparação dos dados

Baixe o dataset original STARCOP pela fonte oficial:

- [Dataset completo no Zenodo](https://doi.org/10.5281/zenodo.7863343), recomendado
  para `starcop_raw`;
- [STARCOP_mini.zip no Google Drive](https://drive.google.com/uc?id=1Qw96Drmk2jzBYSED0YPEUyuc2DnBechl),
  para o subconjunto `starcop_mini`.

Depois de extrair os arquivos, a pasta `data/` deve ficar organizada assim:

```text
data/
├── starcop_mini/
│   ├── train_mini10.csv
│   ├── test_mini10.csv
│   └── <pastas das cenas e arquivos GeoTIFF>
├── starcop_raw/
│   ├── train.csv
│   ├── test.csv
│   ├── STARCOP_test/
│   ├── STARCOP_train_easy/
│   └── STARCOP_train_remaining_part*/
└── processed/
    ├── starcop_mini/
    └── starcop_raw/
```

Se o arquivo compactado criar uma pasta externa, como `STARCOP_mini/`, mova ou
renomeie essa pasta para `data/starcop_mini/`. Não deixe uma camada adicional
como `data/starcop_mini/STARCOP_mini/`.

Os scripts de treinamento e avaliação utilizam os arquivos processados em
`data/processed/<dataset>/`, incluindo as tabelas de patches e os splits. Esses
arquivos devem ser gerados pela preparação do dataset antes da execução dos
experimentos.

Comandos auxiliares:

```bash
# pré-computa os caches dos três splits do starcop_mini
make precompute-cache

# escolhe o dataset e os splits explicitamente
make precompute-cache PRECOMPUTE_CACHE_ARGS="dataset=starcop_raw splits=train,val"

# gera os manifestos R2
make build-r2-manifest

# valida o contrato completo contra dados reais
make confirm-raw
make confirm-raw CONFIRM_RAW_ARGS="dataset=starcop_mini"
```

As mesmas rotinas podem ser executadas diretamente:

```bash
uv run python -m data.precompute_patch_cache dataset=starcop_raw splits=train,val
uv run python -m data.build_r2_manifest
uv run python -m data.confirm_raw dataset=starcop_mini
```

### Treinamento, avaliação e figuras

Os comandos do `Makefile` são a forma recomendada de executar o projeto. Cada
comando aceita argumentos pelo `Makefile`, mantendo os defaults usados nas
execuções principais:

```bash
# treinamento padrão de E1 no starcop_mini
make train

# treinamento de outra arquitetura/configuração
make train TRAIN_ARGS="architecture=E2 dataset=starcop_raw tier=raw-full max_epochs=50"

# treinamento com recuperação automática após falha de GPU
make train-recovery
make train-recovery RECOVERY_ARGS="architecture=E3 dataset=starcop_raw tier=raw-full max_attempts=5"

# avaliação padrão de E1 no starcop_mini
make evaluate

# avaliação com parâmetros diferentes
make evaluate EVALUATE_ARGS="architecture=E2 dataset=starcop_raw tier=raw-full device=cpu"

# avaliação por cena completa ou por patches
make evaluate-scenes
make evaluate-scenes EVALUATE_SCENES_ARGS="architecture=E2 checkpoint_tier=raw-full eval_tier=raw-full mode=patches device=cpu"

# gera todas as figuras do projeto
make figures

# gera apenas uma figura
make figure-architecture
make figure-pr-curves
make figure-qualitative
```

Os comandos equivalentes, para execução direta com parâmetros, são:

```bash
uv run python -m training.train architecture=E2 dataset=starcop_raw tier=raw-full
uv run python -m training.train_with_recovery architecture=E3 dataset=starcop_raw tier=raw-full max_attempts=5
uv run python -m evaluation.evaluate architecture=E2 dataset=starcop_raw tier=raw-full device=cpu
uv run python -m evaluation.evaluate_scenes architecture=E2 checkpoint_tier=raw-full eval_tier=raw-full mode=patches device=cpu

uv run python -m visualization.architecture_diagram
uv run python -m visualization.eda
uv run python -m visualization.class_balance
uv run python -m data.data_quality
uv run python -m visualization.pr_curve_plots
uv run python -m visualization.training_curves
uv run python -m visualization.threshold_effect_plot
uv run python -m visualization.efficiency_plot
uv run python -m visualization.qualitative_predictions
```

Para as rotinas sem argumentos, o `Makefile` oferece alvos equivalentes:
`make data-quality`, `make calibrate-thresholds` e `make idle-power`. Seus
comandos diretos correspondentes são:

```bash
uv run python -m data.data_quality
uv run python -m evaluation.threshold_calibration
uv run python -m utils.power_meter 30
```

O dispositivo é escolhido automaticamente pelo treinamento e pela avaliação,
priorizando CUDA quando disponível. Para forçar CPU ou CUDA, informe `device`
nos argumentos do comando correspondente.

### MLflow

O histórico local dos experimentos está em `mlflow.db`, com os artefatos
versionados em `mlruns/1/`. Para abrir a interface local:

```bash
uv run mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db
```

As rotinas de visualização consultam esse histórico sem exigir novas execuções.

## Qualidade e testes

Os comandos principais de validação são:

```bash
make test
make coverage
make lint
make docstring-coverage
make mutation
```

Também é possível remover os arquivos gerados sem apagar os marcadores `.gitkeep`:

```bash
make clean-generated
```

A suíte de testes fica junto aos domínios correspondentes em
`src/<domínio>/__tests__/`.

## Estrutura do projeto

```text
.
├── configs/
│   ├── data.yaml
│   └── dataset/                 # configurações dos datasets
├── data/
│   ├── manifests/               # manifestos gerados
│   └── ...                       # datasets baixados localmente, ignorados pelo Git
├── figures/                     # figuras geradas, ignoradas pelo Git
├── checkpoints/                 # checkpoints gerados, ignorados pelo Git
├── logs/                        # logs gerados, ignorados pelo Git
├── mlflow.db                    # histórico local do MLflow
├── mlruns/1/                    # artefatos versionados dos experimentos
├── patch_cache/                 # caches gerados, ignorados pelo Git
├── run_state/                   # estados de retomada, ignorados pelo Git
├── scene_scores/                # resultados por cena, ignorados pelo Git
├── src/
│   ├── data/                    # datasets, preprocessing e manifestos
│   ├── evaluation/              # métricas, avaliação e protocolos
│   ├── models/                  # arquiteturas e funções de perda
│   ├── training/                # treinamento e recuperação de execuções
│   ├── utils/                   # utilitários, incluindo medição de energia
│   └── visualization/           # figuras e análises exploratórias
├── Makefile
├── pyproject.toml
└── uv.lock
```
