"""HTML Feature Extraction Functions for Phishing Detection.

This module contains individual feature extraction functions that analyze HTML content
to identify characteristics useful for distinguishing between legitimate and phishing
websites. Features are categorized into two types:

1. Binary Features: Return 1 if the feature is present, 0 otherwise
   Examples: has_title, has_password, has_email_input

2. Quantitative Features: Return integer counts of specific HTML elements
   Examples: number_of_inputs, number_of_images, length_of_title

These features capture patterns commonly found in phishing websites, such as:
    - Presence of password/email input fields (credential harvesting)
    - Hidden form elements (concealing malicious code)
    - Unusual numbers of scripts or form elements
    - Missing or suspiciously short titles

All functions accept a BeautifulSoup object as input and return either an integer
(0 or 1 for binary, counts for quantitative features).
"""

from bs4 import BeautifulSoup

# --- Binary Features ---
def has_title(soup):
    """Check if the webpage has a non-empty title.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if a title exists and contains text, 0 otherwise.
    
    Note:
        Phishing sites sometimes lack titles or have suspicious titles.
    """
    if soup.title and soup.title.text:
        return 1
    return 0

def has_input(soup):
    """Check if the webpage contains any input elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if input elements exist, 0 otherwise.
    
    Note:
        Input elements are common in phishing sites for credential harvesting.
    """
    if len(soup.find_all("input")) > 0:
        return 1
    return 0

def has_button(soup):
    """Check if the webpage contains any button elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if button elements exist, 0 otherwise.
    """
    if len(soup.find_all("button")) > 0:
        return 1
    return 0

def has_image(soup):
    """Check if the webpage contains any image elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if img elements exist, 0 otherwise.
    """
    if len(soup.find_all("img")) > 0:
        return 1
    return 0

def has_submit(soup):
    """Check if the webpage contains submit buttons.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if any input element has type="submit", 0 otherwise.
    
    Note:
        Submit buttons are often present in phishing forms.
    """
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
    return 0

def has_link(soup):
    """Check if the webpage contains link elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if link elements exist, 0 otherwise.
    
    Note:
        Link elements in <head> reference stylesheets, icons, etc.
    """
    if len(soup.find_all("link")) > 0:
        return 1
    return 0

def has_password(soup):
    """Check if the webpage contains password input fields.
    
    Searches for input elements with type="password" or name/id="password".
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if password fields exist, 0 otherwise.
    
    Note:
        Password fields are strong indicators of credential harvesting attempts
        in phishing sites.
    """
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") == "password" or 
            input_tag.get("name") == "password" or 
            input_tag.get("id") == "password"):
            return 1
    return 0

def has_email_input(soup):
    """Check if the webpage contains email input fields.
    
    Searches for input elements with type="email" or name/id="email".
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if email fields exist, 0 otherwise.
    
    Note:
        Email fields combined with password fields are common in phishing login forms.
    """
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") == "email" or 
            input_tag.get("name") == "email" or 
            input_tag.get("id") == "email"):
            return 1
    return 0

def has_hidden_element(soup):
    """Check if the webpage contains hidden input fields.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if hidden input fields exist, 0 otherwise.
    
    Note:
        Hidden fields can be used in phishing sites to conceal malicious data
        or track victims.
    """
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden":
            return 1
    return 0

def has_audio(soup):
    """Check if the webpage contains audio elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if audio elements exist, 0 otherwise.
    """
    if len(soup.find_all("audio")) > 0:
        return 1
    return 0

def has_video(soup):
    """Check if the webpage contains video elements.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: 1 if video elements exist, 0 otherwise.
    """
    if len(soup.find_all("video")) > 0:
        return 1
    return 0

# --- Quantitative Features ---
# These functions return integer counts of specific HTML elements
def number_of_inputs(soup):
    """Count the total number of input elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of input elements.
    
    Note:
        A high number of input fields may indicate a phishing form.
    """
    return len(soup.find_all("input"))

def number_of_buttons(soup):
    """Count the total number of button elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of button elements.
    """
    return len(soup.find_all("button"))

def number_of_images(soup):
    """Count the total number of images in the webpage.
    
    Includes both <img> tags and meta tags with image type/name.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Total count of image elements and image meta tags.
    
    Note:
        Some phishing sites use meta tags to reference images.
    """
    image_tags = len(soup.find_all("img"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") == "image" or meta.get("name") == "image":
            count += 1
    return image_tags + count

def number_of_option(soup):
    """Count the total number of option elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of option elements (typically within select dropdowns).
    """
    return len(soup.find_all("option"))

def number_of_list(soup):
    """Count the total number of list item elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <li> elements.
    """
    return len(soup.find_all("li"))

def number_of_th(soup):
    """Count the total number of table header cells in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <th> elements.
    """
    return len(soup.find_all("th"))

def number_of_tr(soup):
    """Count the total number of table rows in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <tr> elements.
    """
    return len(soup.find_all("tr"))

def number_of_href(soup):
    """Count the number of link elements with href attributes.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <link> elements that have an href attribute.
    
    Note:
        Counts links in <head> (stylesheets, icons, etc.), not <a> hyperlinks.
    """
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count

def number_of_paragraph(soup):
    """Count the total number of paragraph elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <p> elements.
    
    Note:
        Phishing sites may have fewer paragraphs than legitimate informational sites.
    """
    return len(soup.find_all("p"))

def number_of_script(soup):
    """Count the total number of script elements in the webpage.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Count of <script> elements.
    
    Note:
        An unusually high number of scripts might indicate malicious behavior.
    """
    return len(soup.find_all("script"))

def length_of_title(soup):
    """Calculate the character length of the webpage title.
    
    Args:
        soup (BeautifulSoup): Parsed HTML content.
    
    Returns:
        int: Length of the title text after stripping whitespace, or 0 if no title exists.
    
    Note:
        Suspiciously short or missing titles may indicate phishing sites.
    """
    if soup.title and soup.title.text:
        return len(soup.title.text.strip())
    return 0