# Tokenización con Transformers

Proyecto de ejemplo para explorar cómo un tokenizador preentrenado transforma texto en tokens e IDs usando `transformers`.

## Requisitos

- Python 3.10+ (recomendado 3.12)
- Entorno virtual (venv)

## Instalación

1) Crear y activar el entorno virtual:

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

2) Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

### Scripts

```bash
python tokenization.py
```

Otros scripts disponibles:

```bash
python embeddings.py
python pcaVisualization.py
python semanticAnalysis.py
python semanticTopology.py
```

### Notebooks

Abrir y ejecutar los notebooks:

- `tokenizacion_tutorial.ipynb`
- `embeddings_tutorial.ipynb`
- `pca_visualization_tutorial.ipynb`
- `similarity_tutorial.ipynb`
- `semantic_topology_tutorial.ipynb`

## Estructura

- `tokenization.py`: script principal de tokenización.
- `embeddings.py`: generación de embeddings de palabras y frases.
- `pcaVisualization.py`: visualización 3D con PCA.
- `semanticAnalysis.py`: matriz de similitud y búsqueda de vecinos.
- `semanticTopology.py`: topología semántica en 3D.
- `tokenizacion_tutorial.ipynb`: tokenización paso a paso.
- `embeddings_tutorial.ipynb`: embeddings de palabras y oraciones.
- `pca_visualization_tutorial.ipynb`: PCA 3D de embeddings.
- `similarity_tutorial.ipynb`: matriz de similitud coseno.
- `semantic_topology_tutorial.ipynb`: conexiones al vecino más similar.
- `requirements.txt`: dependencias del proyecto.

## Notas

En Windows puede aparecer una advertencia de `huggingface_hub` sobre symlinks. No afecta la ejecución.
