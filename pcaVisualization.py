from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


# --- BLOQUE DE CONFIGURACIÓN (Necesario para generar los datos) ---
print("Cargando modelo y generando embeddings (esto puede tardar unos segundos)...")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)

def get_embedding(text):
    encoded_input = tokenizer([text], padding = True, truncation = True, return_tensors = 'pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p = 2, dim = 1)
    return sentence_embedding[0].numpy()

example_words = [
    "cat", "dog", "lion",
    "red", "blue", "green",
    "code", "python", "java",
    "happy", "sad", "angry",
    "pizza", "hamburger", "pasta"
]

# Generamos la matriz de datos
embeddings_matrix_words = [get_embedding(word) for word in example_words]
# ------------------------------------------------------------------


print("\n" + "="*60)
print("PARTE 3: VISUALIZACION (APROXIMADA) CON PCA")
print("="*60)

# 1. Reducción de dimensionalidad
pca = PCA(n_components = 3)
embeddings_3d = pca.fit_transform(embeddings_matrix_words)


# 2. Análisis de varianza (¿Qué tanta información perdimos?)
print("\n" + "-" * 50)
print(f"Varianza explicada por cada componente:\n")

for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var_ratio:.3f} ({var_ratio*100:.1f}%)")

print(f"\nVarianza total explicada: {pca.explained_variance_ratio_.sum():.3f} ({pca.explained_variance_ratio_.sum()*100:.1f}%)")
print("-" * 50)


# 3. Categorización para colorear el gráfico
categorias = (
    ['Animales'] * 3 + 
    ['Colores'] * 3 + 
    ['Programación'] * 3 + 
    ['Emociones'] * 3 + 
    ['Comida'] * 3
)


# 4. DataFrame para graficar
df_visualization = pd.DataFrame({
    'word': example_words,
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'category': categorias
})


# 5. Gráfico 3D
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection = '3d')


# Mapeo de colores simple para diferenciar categorías
colors = {'Animales':'brown', 'Colores':'green', 'Programación':'blue', 'Emociones':'purple', 'Comida':'orange'}
c_map = df_visualization['category'].map(colors)

ax.scatter(df_visualization['x'], df_visualization['y'], df_visualization['z'], c = c_map, marker = 'o', s = 100)

for i, word in enumerate(df_visualization['word']):
    ax.text(df_visualization['x'][i], df_visualization['y'][i], df_visualization['z'][i], word)

plt.title('Visualización 3D de Embeddings con PCA')
plt.show()