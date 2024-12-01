import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from metaphone import doublemetaphone
account_booking_df = pd.read_csv('account_booking_train.csv')
external_parties_df = pd.read_csv('external_parties_train.csv')
# Combine the tables with merge using 
combined_data = pd.merge(account_booking_df, external_parties_df, on='transaction_reference_id', how='inner')
#Null count of each variables to know which ones are more relevant to analyze
combined_data.isnull().sum()
irrelevant_cols_external = ['party_info_unstructured', 'parsed_address_unit', 'parsed_address_state', 'parsed_address_country']
external_parties_df.drop(columns=irrelevant_cols_external, inplace=True, errors='ignore')

## Accounts booking data
duplicate_ids = account_booking_df[account_booking_df.duplicated(subset='transaction_reference_id', keep=False)]
account_booking_df = account_booking_df[~account_booking_df['transaction_reference_id'].isin(duplicate_ids['transaction_reference_id'])]

irrelevant_cols_booking = ['debit_credit_indicator']
account_booking_df.drop(columns=irrelevant_cols_booking, inplace=True, errors='ignore')

# Merge the two dataframes

merged_df = pd.merge(external_parties_df, account_booking_df, on='transaction_reference_id', how='inner')

merged_df.to_csv('merged_data_cleaned.csv', index=False)

honorifics = ['Mr. ', 'Ms. ', 'Mrs. ', 'Miss ', 'Dr. ', 'Prof. ', 'Rev. ', 'Hon. ', 'mr. ', 'ms. ', 'mrs. ', 'miss ', 'dr. ', 'prof. ', 'rev. ', 'hon. ']
for honorific in honorifics:
    merged_df['parsed_name'] = merged_df['parsed_name'].str.replace(honorific, '')


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

merged_df = delete_duplicates(merged_df)

from metaphone import doublemetaphone

def apply_metaphone(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):  
        return np.nan  
    if pd.isna(value) or not isinstance(value, str):  
        return np.nan
    return doublemetaphone(value.strip().lower())[0] 

def split_and_metaphone(full_name):
    """
    Divide un nombre completo en sus partes (palabras), aplica Metaphone a cada parte,
    y las junta nuevamente con espacios.
    """
    if pd.isna(full_name) or not isinstance(full_name, str):  
        return np.nan

    parts = full_name.strip().split()

    return ' '.join(apply_metaphone(part) for part in parts)

merged_df['parsed_name'] = merged_df['parsed_name'].apply(split_and_metaphone)

print(merged_df['parsed_name'].head())

merged_df['parsed_address_street_name'] = merged_df['parsed_address_street_name'].apply(split_and_metaphone)

print(merged_df['parsed_address_street_name'].head())

merged_df.to_csv('merged_data_cleaned.csv', index=False)
# Función para calcular similitudes con TF-IDF
# Reemplazar valores faltantes de direcciones
merged_df['parsed_address_street_name'].fillna(merged_df['parsed_address_city'].fillna('MISSING'), inplace=True)
def calculate_similarity(df, column1, column2):
    vectorizer = TfidfVectorizer()
    
    # Rellenamos los valores faltantes con un valor temporal
    df[column1].fillna('MISSING', inplace=True)
    df[column2].fillna('MISSING', inplace=True)
    
    # Combinamos las dos columnas en una nueva
    combined_column = df[column1] + " " + df[column2]
    
    # Creamos la matriz TF-IDF
    tfidf_matrix = vectorizer.fit_transform(combined_column)
    
    # Calculamos la matriz de similitud del coseno
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Opcional: Penaliza similitud para valores originalmente faltantes
    missing_mask = df[column1].isna().values | df[column2].isna().values
    cosine_sim[missing_mask, :] = 0
    cosine_sim[:, missing_mask] = 0
    
    return cosine_sim


# Función de blocking
def create_block_key(value, prefix_length=5):
    if pd.isna(value):  # Maneja valores faltantes
        return 'MISSING'
    return value[:prefix_length].lower()

# Aplica el bloqueo en la columna deseada
merged_df['block'] = merged_df['parsed_name'].apply(create_block_key, prefix_length=3)

def create_combined_block(row):
    # Combina prefijos de varias columnas
    name_key = create_block_key(row['parsed_name'], prefix_length=3)
    address_key = create_block_key(row['parsed_address_street_name'], prefix_length=3)
    return f"{name_key}_{address_key}"

merged_df['block'] = merged_df.apply(create_combined_block, axis=1)
blocks = merged_df.groupby('block')


# Procesar cada bloque


merged_df['parsed_address_street_name'].fillna(
merged_df['parsed_address_city'].fillna('MISSING'), inplace=True
)

results = []

# En el bucle de procesamiento de bloques, en lugar de calcular solo la similitud de `parsed_name`, ahora usaremos las dos columnas
for block_name, block_data in blocks:
    print(f"Processing block: {block_name}")
    
    if len(block_data) < 2:
        block_data['external_id'] = range(len(results), len(results) + len(block_data))
        results.append(block_data)
        continue

    # Similaridad para 'parsed_name' y 'parsed_address_street_name'
    name_and_address_similarity = calculate_similarity(block_data, 'parsed_name', 'parsed_address_street_name')
    
    # Asegúrate de que los valores de la matriz de similitud estén en el rango [0, 1]
    name_and_address_similarity = np.clip(name_and_address_similarity, 0, 1)
    
    # Convertimos la matriz de similitud en distancias
    distance_matrix = 1 - name_and_address_similarity
    
    # Aplicamos DBSCAN
    db = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
    clusters = db.fit_predict(distance_matrix)
    
    # Asignamos clusters como 'external_id'
    block_data['external_id'] = clusters + len(results)  # Evita conflictos entre bloques
    results.append(block_data)


# Combinar los resultados de todos los bloques
final_df = pd.concat(results, ignore_index=True)
cluster_sizes = final_df['external_id'].value_counts()
valid_clusters = cluster_sizes[cluster_sizes > 1].index

# Filtramos las filas que pertenecen a esos clusters
filtered_df = final_df[final_df['external_id'].isin(valid_clusters)]
filtered_df = filtered_df[['external_id', 'transaction_reference_id']]

filtered_df.to_excel('filtered_clusters.xlsx', index=False)

