import pandas as pd
import numpy as np
from metaphone import doublemetaphone

# Cargar los datos
external_parties_path = 'external_parties_train.csv'
account_booking_path = 'account_booking_train.csv'

external_parties_df = pd.read_csv(external_parties_path)
account_booking_df = pd.read_csv(account_booking_path)

# Función para aplicar Metaphone
def apply_metaphone(value):
    """Aplica Metaphone a un valor para obtener una representación fonética."""
    if pd.isna(value) or not isinstance(value, str):
        return np.nan
    return doublemetaphone(value.strip().lower())[0]  

if 'parsed_name' in external_parties_df.columns:
    external_parties_df['parsed_name_metaphone'] = external_parties_df['parsed_name'].apply(apply_metaphone)

if 'parsed_address_clean' in external_parties_df.columns:
    external_parties_df['parsed_address_clean_metaphone'] = external_parties_df['parsed_address_clean'].apply(apply_metaphone)

for column in external_parties_df.columns:
    if "name" in column.lower() or "address" in column.lower():
        new_col = f"{column}_metaphone"
        if new_col not in external_parties_df.columns:
            external_parties_df[new_col] = external_parties_df[column].apply(apply_metaphone)


irrelevant_cols_external = ['party_info_unstructured', 'parsed_address_unit', 'parsed_address_state', 'parsed_address_country']
external_parties_df.drop(columns=irrelevant_cols_external, inplace=True, errors='ignore')

duplicate_ids = account_booking_df[account_booking_df.duplicated(subset='transaction_reference_id', keep=False)]
account_booking_df = account_booking_df[~account_booking_df['transaction_reference_id'].isin(duplicate_ids['transaction_reference_id'])]

irrelevant_cols_booking = ['debit_credit_indicator', 'transaction_amount', 'transaction_currency']
account_booking_df.drop(columns=irrelevant_cols_booking, inplace=True, errors='ignore')

merged_df = pd.merge(external_parties_df, account_booking_df, on='transaction_reference_id', how='inner')

merged_df.to_excel('merged_data_cleaned.xlsx', index=False)


