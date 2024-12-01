import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

# Función para calcular similitudes con TF-IDF
def calculate_similarity(df, column):
    vectorizer = TfidfVectorizer()
    filled_column = df[column].fillna('MISSING')  # Rellena missings temporalmente
    tfidf_matrix = vectorizer.fit_transform(filled_column)
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Opcional: Penaliza similitud para valores originalmente faltantes
    missing_mask = df[column].isna().values
    cosine_sim[missing_mask, :] = 0
    cosine_sim[:, missing_mask] = 0
    
    return cosine_sim

# Función de bloqueo (para crear claves de bloqueo en base a los primeros caracteres)
def create_block_key(value, prefix_length=5):
    if pd.isna(value):  # Maneja valores faltantes
        return 'MISSING'
    return value[:prefix_length].lower()

# Función para crear un bloque combinado que utiliza nombre y dirección
def create_combined_block(row):
    name_key = create_block_key(row['parsed_name'], prefix_length=3)
    address_key = create_block_key(row['parsed_address_street_name'], prefix_length=3)
    return f"{name_key}_{address_key}"

# Función para eliminar duplicados de nombres
def delete_duplicates(name):
    def remove_duplicates(text):
        words = text.split()
        seen = set()
        result = []
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                result.append(word)
        return ' '.join(result)
    
    name['parsed_name'] = name['parsed_name'].apply(remove_duplicates)
    return name

# Procesamiento de datos y carga de archivos
account_booking_df = pd.read_csv('account_booking_train.csv')
external_parties_df = pd.read_csv('external_parties_train.csv')

# Combinación de los dos DataFrames
combined_data = pd.merge(account_booking_df, external_parties_df, on='transaction_reference_id', how='inner')

# Eliminar columnas irrelevantes
irrelevant_cols_external = ['party_info_unstructured', 'parsed_address_unit', 'parsed_address_state', 'parsed_address_country']
external_parties_df.drop(columns=irrelevant_cols_external, inplace=True, errors='ignore')

irrelevant_cols_booking = ['debit_credit_indicator']
account_booking_df.drop(columns=irrelevant_cols_booking, inplace=True, errors='ignore')

# Merge
merged_df = pd.merge(external_parties_df, account_booking_df, on='transaction_reference_id', how='inner')

# Limpiar y procesar nombres
honorifics = ['Mr. ', 'Ms. ', 'Mrs. ', 'Miss ', 'Dr. ', 'Prof. ', 'Rev. ', 'Hon. ', 'mr. ', 'ms. ', 'mrs. ', 'miss ', 'dr. ', 'prof. ', 'rev. ', 'hon. ']
for honorific in honorifics:
    merged_df['parsed_name'] = merged_df['parsed_name'].str.replace(honorific, '')

merged_df = delete_duplicates(merged_df)

# Aplicar Metaphone a los nombres y direcciones
from metaphone import doublemetaphone

def apply_metaphone(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):  
        return np.nan  
    if pd.isna(value) or not isinstance(value, str):  
        return np.nan
    return doublemetaphone(value.strip().lower())[0] 

def split_and_metaphone(full_name):
    if pd.isna(full_name) or not isinstance(full_name, str):  
        return np.nan

    parts = full_name.strip().split()
    return ' '.join(apply_metaphone(part) for part in parts)

merged_df['parsed_name'] = merged_df['parsed_name'].apply(split_and_metaphone)
merged_df['parsed_address_street_name'] = merged_df['parsed_address_street_name'].apply(split_and_metaphone)

# Crear claves de bloqueo y combinarlas
merged_df['block'] = merged_df.apply(create_combined_block, axis=1)

# Agrupar por bloque
blocks = merged_df.groupby('block')

# Reemplazar valores faltantes de direcciones
merged_df['parsed_address_street_name'].fillna(merged_df['parsed_address_city'].fillna('MISSING'), inplace=True)

# Iniciar el procesamiento de bloques
results = []

for block_name, block_data in blocks:
    print(f"Processing block: {block_name}")
    
    if len(block_data) < 2:
        # Si el bloque tiene solo un registro, lo marcamos como un singleton
        block_data['external_id'] = range(len(results), len(results) + len(block_data))
        results.append(block_data)
        continue

    # Similaridad para nombres y direcciones
    name_similarity = calculate_similarity(block_data, 'parsed_name')
    
    # Convertimos la matriz de similitud en distancias
    distance_matrix = 1 - np.clip(name_similarity, 0, 1)  # Convertir similitudes a distancias
    
    # Aplicamos DBSCAN con la matriz de distancias
    db = DBSCAN(eps=0.05, min_samples=100, metric='precomputed')  # Ajustar eps y min_samples según sea necesario
    clusters = db.fit_predict(distance_matrix)
    
    # Asignamos clusters como external_id
    block_data['external_id'] = clusters + len(results)  # Evita conflictos entre bloques
    results.append(block_data)

# Combinar los resultados de todos los bloques
final_df = pd.concat(results, ignore_index=True)

# Visualizar el tamaño de cada cluster
cluster_sizes = final_df['external_id'].value_counts()

# plt.figure(figsize=(12, 6))
# sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
# plt.title('Distribución de Clusters')
# plt.xlabel('Cluster ID')
# plt.ylabel('Tamaño del Cluster')
# plt.show()
# Aplicamos DBSCAN con la matriz de distancias

# Comprobar si hay más de un cluster
unique_clusters = len(set(clusters))

if unique_clusters > 1:
    silhouette = silhouette_score(distance_matrix, clusters, metric='precomputed')
    print(f"Silhouette Score: {silhouette}")
else:
    print("Solo se ha generado un cluster. No se puede calcular el Silhouette Score.")

