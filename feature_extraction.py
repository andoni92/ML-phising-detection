"""Feature Extraction Module for Phishing Detection.

This module provides utility functions to extract features from website HTML content
that are used by machine learning models to classify websites as legitimate or phishing.

The module creates feature vectors containing both binary features (presence/absence of
HTML elements) and quantitative features (counts of specific elements) that have been
identified as useful indicators for phishing detection.

Functions:
    create_soup: Parse HTML text into a BeautifulSoup object
    create_vector: Extract a complete feature vector from parsed HTML
"""

from bs4 import BeautifulSoup
import features 

def create_soup(text):
    """Parse HTML text into a BeautifulSoup object for feature extraction.
    
    Args:
        text (str): Raw HTML content as a string.
    
    Returns:
        BeautifulSoup: A parsed BeautifulSoup object that can be used to
                      extract HTML elements and features.
    
    Note:
        Uses the 'html.parser' which is included with Python's standard library.
    """
    return BeautifulSoup(text, "html.parser")

def create_vector(soup):
    """Extract a complete feature vector from parsed HTML content.
    
    This function extracts 22 features from the HTML that are used by machine
    learning models to classify websites. Features include both binary indicators
    (e.g., has_title, has_password) and quantitative counts (e.g., number_of_inputs).
    
    Args:
        soup (BeautifulSoup): A BeautifulSoup object containing parsed HTML.
    
    Returns:
        list: A feature vector containing 22 elements in the following order:
            Binary features (0 or 1):
                1. has_title - Presence of a title tag
                2. has_input - Presence of input elements
                3. has_button - Presence of button elements
                4. has_image - Presence of image elements
                5. has_submit - Presence of submit buttons
                6. has_link - Presence of link elements
                7. has_password - Presence of password input fields
                8. has_email_input - Presence of email input fields
                9. has_hidden_element - Presence of hidden input fields
                10. has_audio - Presence of audio elements
                11. has_video - Presence of video elements
            
            Quantitative features (integer counts):
                12. number_of_inputs - Count of input elements
                13. number_of_buttons - Count of button elements
                14. number_of_images - Count of image elements
                15. number_of_option - Count of option elements
                16. number_of_list - Count of list items
                17. number_of_th - Count of table header cells
                18. number_of_tr - Count of table rows
                19. number_of_href - Count of links with href attributes
                20. number_of_paragraph - Count of paragraph elements
                21. number_of_script - Count of script elements
                22. length_of_title - Character length of the title text
    
    Example:
        >>> from bs4 import BeautifulSoup
        >>> html = '<html><head><title>Test</title></head><body><input type="text"/></body></html>'
        >>> soup = BeautifulSoup(html, 'html.parser')
        >>> vector = create_vector(soup)
        >>> len(vector)
        22
    """
    return [
        features.has_title(soup),
        features.has_input(soup),
        features.has_button(soup),
        features.has_image(soup),
        features.has_submit(soup),
        features.has_link(soup),
        features.has_password(soup),
        features.has_email_input(soup),
        features.has_hidden_element(soup),
        features.has_audio(soup),
        features.has_video(soup),
        features.number_of_inputs(soup),
        features.number_of_buttons(soup),
        features.number_of_images(soup),
        features.number_of_option(soup),
        features.number_of_list(soup),
        features.number_of_th(soup),
        features.number_of_tr(soup),
        features.number_of_href(soup),
        features.number_of_paragraph(soup),
        features.number_of_script(soup),
        features.length_of_title(soup)
    ]