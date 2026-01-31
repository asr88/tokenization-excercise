from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch


# --- BLOQUE DE CONFIGURACIÓN ---
print("Generando datos para análisis de similitud...")

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
print("PARTE 4: MATRIZ DE SIMILITUD Y BÚSQUEDA")
print("="*60)

# 1. Cálculo de matriz de similitud
similarity_matrix = cosine_similarity(embeddings_matrix_words)
df_similarity = pd.DataFrame(similarity_matrix, index = example_words, columns = example_words)

# 2. Visualización Heatmap
plt.figure(figsize = (10, 8))
sns.heatmap(df_similarity, annot = True, cmap = "coolwarm", fmt = ".2f")
plt.title("Matriz de Similitud Coseno")
plt.show()

# 3. Función de Búsqueda de Vecinos
def encontrar_mas_similares(reference_word, top_k = 2):
    if reference_word not in example_words:
        print(f"\nPalabra '{reference_word}' no encontrada")
        return

    index_ref = example_words.index(reference_word)
    similarities = similarity_matrix[index_ref]

    # Ordena índices de mayor a menor similitud
    sorted_index = np.argsort(similarities)[::-1][1:top_k+1]

    print(f"\nPalabras más similares a: '{reference_word}'")
    print("-" * 40)

    for i, index in enumerate(sorted_index, 1):
        similar_word = example_words[index]
        similarity = similarities[index]
        print(f"{i}. '{similar_word}' (similitud: {similarity:.3f})")

# 4. Ejecución de la búsqueda
print("\n" + "="*60)
print("PARTE 5: ANALISIS DETALLADO")
print("="*60)

analysis_examples = ["cat", "red", "python", "happy", "pizza"]
for example in analysis_examples:
    encontrar_mas_similares(example)