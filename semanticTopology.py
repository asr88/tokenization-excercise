from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
import torch

# --- BLOQUE DE CONFIGURACIÓN ---
print("Preparando visualización avanzada...")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    return sentence_embedding[0].numpy()

example_words = [
    "cat", "dog", "lion", "red", "blue", "green",
    "code", "python", "java", "happy", "sad", "angry",
    "pizza", "hamburger", "pasta"
]

embeddings_matrix_words = [get_embedding(word) for word in example_words]
# ------------------------------------------------------------------

print("\n" + "="*60)
print("PARTE 6: VISUALIZACIÓN DE RELACIONES EN EL ESPACIO")
print("="*60)

# 1. Calcular PCA y Matriz de Similitud
pca = PCA(n_components = 3)
embeddings_3d = pca.fit_transform(embeddings_matrix_words)
similarity_matrix = cosine_similarity(embeddings_matrix_words)

# 2. Configurar gráfico 3D
fig = plt.figure(figsize = (10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Dibujar puntos
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], color = 'black', s = 20)

for i, word in enumerate(example_words):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], word)

# 4. Dibujar conexiones al vecino más cercano
for i in range(len(example_words)):
    
    # Buscar el más similar (excluyéndose a sí mismo)
    similarities = similarity_matrix[i].copy()
    similarities[i] = -1 
    j = np.argmax(similarities)
    
    # Coordenadas de la línea
    xs = [embeddings_3d[i, 0], embeddings_3d[j, 0]]
    ys = [embeddings_3d[i, 1], embeddings_3d[j, 1]]
    zs = [embeddings_3d[i, 2], embeddings_3d[j, 2]]

    # Color según similitud (Amarillo = Muy cerca, Azul = Lejos)
    sim = similarities[j]
    color = plt.cm.viridis(sim)
    ax.plot(xs, ys, zs, color = color, alpha = 0.7)

    # Etiqueta de distancia en la mitad de la línea
    mid_x = (xs[0] + xs[1]) / 2
    mid_y = (ys[0] + ys[1]) / 2
    mid_z = (zs[0] + zs[1]) / 2
    distance = 1 - sim
    ax.text(mid_x, mid_y, mid_z, f"{distance:.2f}", color='black', fontsize=8)

ax.set_title("Topología Semántica (Conexiones al más similar)")
plt.show()