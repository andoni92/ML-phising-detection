"""Data Collection Script for Phishing Detection Dataset.

This script collects and processes website data to create structured datasets for
machine learning model training. It fetches HTML content from URLs, extracts features,
and saves them to CSV files with appropriate labels (0 for legitimate, 1 for phishing).

The script can be configured to process either:
    - Legitimate websites from top-1m.csv (Tranco list)
    - Phishing websites from verified_online.csv (PhishTank verified URLs)

Features extracted include HTML element counts, form elements, and structural properties
that help distinguish between legitimate and phishing websites.

Usage:
    1. Configure the URL_FILENAME, OUTPUT_FILENAME, and LABEL_VALUE variables
    2. Run the script: python data_collector_legit.py
    3. The structured dataset will be saved as a CSV file

Note: SSL verification is disabled for this script to handle sites with certificate issues.
"""

import requests as re
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from bs4 import BeautifulSoup
import pandas as pd
import feature_extraction as fe

# Disable SSL warning messages to keep console output clean
disable_warnings(InsecureRequestWarning)

# --- CONFIGURATION: CHANGE THESE SETTINGS BASED ON THE DATASET TYPE ---
# 
# FOR LEGITIMATE WEBSITES:
# Use the top-1m.csv file containing popular legitimate websites from Tranco
url_filename = r"datasets\top-1m.csv"
output_filename = r"structured_data_legitimate.csv"
label_value = 0  # Label 0 indicates legitimate websites

# FOR PHISHING WEBSITES (Uncomment these lines and comment out the above):
# Use verified_online.csv containing verified phishing URLs from PhishTank
#url_filename = r"datasets\verified_online.csv"
#output_filename = r"structured_data_phishing.csv"
#label_value = 1  # Label 1 indicates phishing websites
# -----------------------------------------------------------------------

def normalize_url(url):
    """Normalize a URL by ensuring it has a proper HTTP/HTTPS protocol prefix.
    
    Some URL lists may contain URLs without the protocol prefix. This function
    adds 'http://' prefix if no protocol is specified.
    
    Args:
        url (str): The URL to normalize. May or may not include protocol prefix.
    
    Returns:
        str: The normalized URL with protocol prefix included.
    
    Examples:
        >>> normalize_url("google.com")
        "http://google.com"
        >>> normalize_url("https://google.com")
        "https://google.com"
    """
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return "http://" + url

def create_structured_data(url_list):
    """Process a list of URLs and extract features to create structured data.
    
    This function iterates through each URL, fetches its HTML content, extracts
    relevant features using the feature_extraction module, and compiles the data
    into a list of feature vectors suitable for machine learning.
    
    The function includes robust error handling to continue processing even if
    individual URLs fail due to connection issues, timeouts, or parsing errors.
    
    Args:
        url_list (list): A list of URL strings to process.
    
    Returns:
        list: A list of feature vectors, where each vector contains:
            - Binary features (has_title, has_input, etc.)
            - Quantitative features (number_of_inputs, number_of_images, etc.)
            - The original URL as the last element
    
    Note:
        - SSL verification is disabled to handle sites with certificate issues
        - Connection timeout is set to 4 seconds per URL
        - Failed requests are logged and skipped without stopping the process
    """
    data_list = []
    for i in range(0, len(url_list)):
        current_url = url_list[i].strip()  # Remove any whitespace or newline characters
        try:
            print(f"Processing {i}: {current_url}")
            
            # Fetch the website with SSL verification disabled and 4-second timeout
            # allow_redirects=True is the default behavior
            response = re.get(current_url, verify=False, timeout=4)
            
            # Check if the HTTP request was successful (status code 200)
            if response.status_code != 200:
                print(f"{i}. HTTP connection was not successful ({response.status_code}) for: {current_url}")
            else:
                # Parse the HTML content
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Extract features using the feature_extraction module
                vector = fe.create_vector(soup)
                
                # Append the URL to the feature vector for reference
                vector.append(str(current_url))
                data_list.append(vector)

        # Catch connection errors, parsing errors, and any other unexpected exceptions
        # Continue processing remaining URLs even if one fails
        except (re.exceptions.RequestException, Exception) as e:
            print(f"{i} --> Error processing {current_url}: {e}")
            continue
            
    return data_list

# Main execution: Load URLs, process them, and save the structured dataset
try:
    # Load the URL list from the CSV file
    df_urls = pd.read_csv(url_filename)
    
    # Handle different CSV formats - some may have a 'url' column header, others may not
    if 'url' in df_urls.columns:
        url_list = df_urls['url'].tolist()
    else:
        # If no header exists, take the first column
        url_list = df_urls.iloc[:, 0].tolist()
    
    # Select the first 1,000 URLs for data collection
    # Adjust this number based on your computational resources and time constraints
    collection_list = url_list[0:1000]
    
    # Normalize all URLs to ensure they have proper protocol prefixes
    collection_list = [normalize_url(url) for url in collection_list]

    # Process URLs and extract features
    data = create_structured_data(collection_list)

    # Define column names matching the features extracted by feature_extraction module
    columns = [
        "has_title", "has_input", "has_button", "has_image", "has_submit", "has_link",
        "has_password", "has_email_input", "has_hidden_element", "has_audio", "has_video",
        "number_of_inputs", "number_of_buttons", "number_of_images", "number_of_option",
        "number_of_list", "number_of_th", "number_of_tr", "number_of_href",
        "number_of_paragraph", "number_of_script", "length_of_title", "URL"
    ]

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data=data, columns=columns)
    
    # Add the label column (0 for legitimate, 1 for phishing)
    df['label'] = label_value
    
    # Save the structured dataset to a CSV file
    df.to_csv(output_filename, index=False)
    print(f"File {output_filename} created successfully.")

except FileNotFoundError:
    print(f"Error: Cannot find the file {url_filename}. Make sure to download it first.")