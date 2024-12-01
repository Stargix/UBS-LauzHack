
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from metaphone import doublemetaphone

account_booking_df = pd.read_csv('account_booking_test.csv')
external_parties_df = pd.read_csv('external_parties_test.csv')

# Combine the tables with merge using 
combined_data = pd.merge(account_booking_df, external_parties_df, on='transaction_reference_id', how='inner')

# Null count of each variables to know which ones are more relevant to analyze
combined_data.isnull().sum()
## External parties data

irrelevant_cols_external = ['party_info_unstructured', 'parsed_address_unit', 'parsed_address_state', 'parsed_address_country', 'parsed_address_street_number', 'party_iban', 'party_phone']
external_parties_df.drop(columns=irrelevant_cols_external, inplace=True, errors='ignore')

## Accounts booking data
duplicate_ids = account_booking_df[account_booking_df.duplicated(subset='transaction_reference_id', keep=False)]
account_booking_df = account_booking_df[~account_booking_df['transaction_reference_id'].isin(duplicate_ids['transaction_reference_id'])]

irrelevant_cols_booking = ['debit_credit_indicator', 'transaction_currency']
account_booking_df.drop(columns=irrelevant_cols_booking, inplace=True, errors='ignore')

# Merge the two dataframes

merged_df = pd.merge(external_parties_df, account_booking_df, on='transaction_reference_id', how='inner')

merged_df.to_csv('merged_data_cleaned.csv', index=False)

# merged_df.to_csv('merged_data_cleaned.csv', index=False)

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



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_similarity(df, column):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[column].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim



def create_block_key(value, prefix_length=3):
    if pd.isna(value):
        return 'MISSING'
    return value[:prefix_length].lower()

def create_combined_block(row):
    name_key = create_block_key(row['parsed_name'])
    address_key = create_block_key(row['parsed_address_street_name'])
    return f"{name_key}_{address_key}"

merged_df['block'] = merged_df.apply(create_combined_block, axis=1)
blocks = merged_df.groupby('block')

print(blocks.size().sort_values(ascending=False).head(10))

results = []

for block_name, block_data in blocks:
    if len(block_data) < 2:
        block_data['external_id'] = range(len(results), len(results) + len(block_data))
        results.append(block_data)
        continue

    name_similarity = calculate_similarity(block_data, 'parsed_name')
    distance_matrix = 1 - name_similarity

    db = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
    clusters = db.fit_predict(distance_matrix)

    block_data['external_id'] = clusters + len(results)
    results.append(block_data)

final_df = pd.concat(results, ignore_index=True)

from sklearn.decomposition import PCA

similarity_matrix = calculate_similarity(final_df, 'parsed_name')
similarity_matrix_clean = np.nan_to_num(similarity_matrix)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(similarity_matrix_clean)

final_df['pca_x'] = reduced_data[:, 0]
final_df['pca_y'] = reduced_data[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=final_df,
    x='pca_x',
    y='pca_y',
    hue='external_id',
    palette='viridis',
    legend=None
)
plt.title('Visualización de Clusters')
plt.show()

cluster_sizes = final_df['external_id'].value_counts()

final_df.to_excel('tests_runs_data.xlsx', index=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
plt.title('Distribución de Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Tamaño del Cluster')
plt.show()
