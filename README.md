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

### Script

```bash
python tokenization.py
```

### Notebook

Abrir y ejecutar el notebook:

- `tokenizacion_tutorial.ipynb`

## Estructura

- `tokenization.py`: script principal de tokenización.
- `tokenizacion_tutorial.ipynb`: notebook con pasos guiados.
- `requirements.txt`: dependencias del proyecto.

## Notas

En Windows puede aparecer una advertencia de `huggingface_hub` sobre symlinks. No afecta la ejecución.
