import requests as re
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe

disable_warnings(InsecureRequestWarning)

# --- CONFIGURACIÓN: CAMBIAR ESTO SEGÚN EL DATASET ---
# PARA LEGÍTIMOS:
url_filename = r"datasets\top-1m.csv"
output_filename = r"structured_data_legitimate.csv"
label_value = 0

# PARA PHISHING (Descomentar y comentar lo de arriba):
#url_filename = r"datasets\verified_online.csv"
#output_filename = r"structured_data_phishing.csv"
#label_value = 1
# ----------------------------------------------------

def normalize_url(url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url

def create_structured_data(url_list):
    data_list = []
    for i in range(0, len(url_list)):
        current_url = url_list[i].strip() # Elimina espacios o saltos de línea
        try:
            print(f"Processing {i}: {current_url}")
            # Agregamos allow_redirects=True (por defecto lo es) y un manejo más robusto
            response = re.get(current_url, verify=False, timeout=4)
            
            if response.status_code != 200:
                print(f"{i}. HTTP connection was not successful ({response.status_code}) for: {current_url}")
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                vector = fe.create_vector(soup)
                vector.append(str(current_url))
                data_list.append(vector)

        # Capturamos errores de parsing, de conexión y cualquier otro error inesperado para no detener el script
        except (re.exceptions.RequestException, Exception) as e:
            print(f"{i} --> Error procesando {current_url}: {e}")
            continue 
    return data_list

# Cargar URLs
# Ajusta 'nrows' o índices para procesar más datos
try:
    df_urls = pd.read_csv(url_filename)
    # Ajuste: Asumimos que la columna se llama 'url'. Si no tiene cabecera, ajusta esto.
    if 'url' in df_urls.columns:
        url_list = df_urls['url'].tolist()
    else:
        # Si no hay header, toma la primera columna
        url_list = df_urls.iloc[:, 0].tolist() 
    
    # Cogemos 1.000 URLs para la recolección
    collection_list = url_list[0:1000] 
    collection_list = [normalize_url(url) for url in collection_list]

    data = create_structured_data(collection_list)

    columns = [
        "has_title", "has_input", "has_button", "has_image", "has_submit", "has_link",
        "has_password", "has_email_input", "has_hidden_element", "has_audio", "has_video",
        "number_of_inputs", "number_of_buttons", "number_of_images", "number_of_option",
        "number_of_list", "number_of_th", "number_of_tr", "number_of_href",
        "number_of_paragraph", "number_of_script", "length_of_title", "URL"
    ]

    df = pd.DataFrame(data=data, columns=columns)
    
    # Etiquetado
    df['label'] = label_value
    
    # Guardar
    df.to_csv(output_filename, index=False)
    print(f"Archivo {output_filename} creado exitosamente.")

except FileNotFoundError:
    print(f"Error: No se encuentra el archivo {url_filename}. Asegúrate de descargarlo.")