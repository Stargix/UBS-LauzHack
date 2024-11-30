from fuzzy import Soundex

# Inicializar Soundex
soundex = Soundex(4)  # Limita a 4 caracteres

def normalize_name(name):
    """Convierte un nombre en una representación fonética utilizando Soundex."""
    if not name or not isinstance(name, str):
        return "UNKNOWN"
    return soundex(name.strip().lower())

name = "Jonny D."
print(normalize_name(name))  # Output: "J530"

import re

def expand_abbreviations(text):
    """Expande abreviaturas comunes en direcciones."""
    if not text or not isinstance(text, str):
        return "UNKNOWN"
    abbreviations = {
        r'\bSt\b': 'Street',
        r'\bAve\b': 'Avenue',
        r'\bRd\b': 'Road',
        r'\bDr\b': 'Drive',
        r'\bBlvd\b': 'Boulevard',
        r'\bLn\b': 'Lane',
        r'\bP\.?O\.?\b': 'Post Office',
    }
    for abbrev, full_form in abbreviations.items():
        text = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
    return text.strip()

address = "123 St. John's Rd"
print(expand_abbreviations(address))  # Output: "123 Street John's Road"

import phonenumbers

def standardize_phone(phone, default_country='US'):
    """Estandariza números telefónicos al formato internacional."""
    try:
        parsed_phone = phonenumbers.parse(phone, default_country)
        if phonenumbers.is_valid_number(parsed_phone):
            return phonenumbers.format_number(parsed_phone, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        pass
    return "INVALID"

phone = "+1 555-123-4567"
print(standardize_phone(phone))  # Output: "+15551234567"

def standardize_postal_code(postal_code):
    """Estandariza códigos postales eliminando espacios y convirtiendo a mayúsculas."""
    if not postal_code or not isinstance(postal_code, str):
        return "UNKNOWN"
    return postal_code.strip().replace(" ", "").upper()

postal_code = "  123 45 "
print(standardize_postal_code(postal_code))  # Output: "12345"

import re
import phonenumbers

def estandarizar_telefono(telefono):
    # Elimina todos los caracteres no numéricos
    telefono = re.sub(r'\D', '', telefono)
    
    # Si el teléfono tiene 10 dígitos, agregar el código de país por defecto
    if len(telefono) == 10:
        telefono = '+1' + telefono
    return telefono

def validar_telefono(telefono):
    try:
        numero = phonenumbers.parse(telefono, None)
        if phonenumbers.is_valid_number(numero):
            return phonenumbers.format_number(numero, phonenumbers.PhoneNumberFormat.E164)
        else:
            return None
    except phonenumbers.phonenumberutil.NumberParseException:
        return None

def procesar_telefono(telefono):
    telefono_estandarizado = estandarizar_telefono(telefono)
    return validar_telefono(telefono_estandarizado)

# Ejemplo de uso
telefono = "+1 (123) 456-7890"
telefono_normalizado = procesar_telefono(telefono)
print(telefono_normalizado)  # Salida: +11234567890

