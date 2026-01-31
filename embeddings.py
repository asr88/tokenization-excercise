from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch


# Cargamos un tokenizador preentrenado para embeddings de oraciones
# Este se usará para convertir texto a tokens y sus IDs correspondientes
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# Función de Mean Pooling
def mean_pooling(model_output, attention_mask):

    """
    Promedia los embeddings de los tokens de una secuencia, ignorando el padding.
    Retorna un único vector por cada oración.
    """

    token_embeddings = model_output[0] # Contiene todos los embeddings de los tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Suma ponderada de embeddings / Cantidad de tokens reales
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)


# Cargamos el modelo preentrenado para generar embeddings de oraciones
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def get_embedding(text):
    
        """
        Esta función obtiene el embedding normalizado de una oración dada
        utilizando el modelo cargado y la función de mean pooling.
        """
    
        # Codifica el texto a tensores de entrada para el modelo
        encoded_input = tokenizer([text], padding = True, truncation = True, return_tensors = 'pt')

        # Desactiva gradientes para eficiencia
        with torch.no_grad(): 
                model_output = model(**encoded_input)
                
        # Aplica mean pooling para obtener el embedding de la oración
        sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normaliza el embedding
        sentence_embedding = F.normalize(sentence_embedding, p = 2, dim = 1)

        return sentence_embedding[0].numpy()


# Lista de palabras de ejemplos para obtener embeddings
example_words = [
    "cat", "dog", "lion",
    "red", "blue", "green",
    "code", "python", "java",
    "happy", "sad", "angry",
    "pizza", "hamburger", "pasta"
]


# Lista de frases de ejemplo para obtener embeddings
example_phrases = [
    "The sun is shining brightly through the window this morning.",
    "She loves to read books about history and ancient civilizations.",
    "It feels so cold outside that my hands are starting to go numb.",
    "They won the game after playing an intense and challenging match."
]


embeddings_words = {}
embeddings_matrix_words = []

print("\n" + "="*50)
print("EMBEDDINGS PARA PALABRAS INDIVIDUALES")
print("="*50)

# Obtenemos y mostramos embeddings para palabras individuales
for word in example_words:
    embedding = get_embedding(word)
    embeddings_words[word] = embedding
    embeddings_matrix_words.append(embedding)
    print(f"\nPalabra: '{word}' → Vector de {len(embedding)} dimensiones")
    print(f"Primeras 5 dimensiones del embedding: {embedding[:5]}")

embeddings_phrases = {}

print("\n" + "="*50)
print("EMBEDDINGS PARA ORACIONES O FRASES")
print("="*50)

# Obtenemos y mostramos embeddings para frases
for phrase in example_phrases:
    embedding = get_embedding(phrase)
    embeddings_phrases[phrase] = embedding
    print(f"\nFrase: '{phrase}' → Vector de {len(embedding)} dimensiones")
    print(f"Primeras 5 dimensiones del embedding: {embedding[:5]}")