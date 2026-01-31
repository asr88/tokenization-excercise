from transformers import AutoTokenizer


# Cargamos un tokenizador preentrenado para embeddings de oraciones
# Este se usará para convertir texto a tokens y sus IDs correspondientes
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def explore_tokenization(text):

    """
    Esta función muestra cómo el tokenizador transforma un texto dado en tokens e IDs,
    imprimiendo información detallada para su análisis.
    """

    print(f"\nAnalizando: '{text}'")
    print("-" * 50)

    # Tokeniza el texto (convierte string a lista de strings/subpalabras)
    tokens = tokenizer.tokenize(text)

    # Codifica el texto a IDs (convierte string a lista de números)
    # Nota: encode agrega automáticamente tokens especiales al inicio y final
    tokens_ids = tokenizer.encode(text)  

    print(f"\nTexto original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Número de tokens: {len(tokens)}")
    print(f"IDs de tokens: {tokens_ids}")

    print("\nCorrespondencia Token → ID:")

    # Excluimos [CLS] (inicio) y [SEP] (fin) al imprimir la correspondencia manual
    # tokens_ids[1:-1] toma los IDs del contenido real
    for i, (token, token_id) in enumerate(zip(tokens, tokens_ids[1:-1])):
        print(f"  {i+1}. '{token}' → {token_id}")

    return tokens, tokens_ids


examples = [
    "The sun is shining brightly through the window this morning",
    "She loves to read books about history and ancient civilizations",
    "It feels so cold outside that my hands are starting to go numb",
    "They won the game after playing an intense and challenging match"
]


# Diccionario para almacenar los resultados de tokenización
tokenization_results = {}


print("\n" + "="*60)
print("PARTE 1: ENTENDIENDO LOS TOKENS")
print("="*60)


# Procesamos cada oración de ejemplo y almacenamos tokens e IDs
for example in examples:
    tokens, tokens_ids = explore_tokenization(example)
    tokenization_results[example] = {"tokens": tokens, "ids": tokens_ids}