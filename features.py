from bs4 import BeautifulSoup

# --- Binary Features ---
def has_title(soup):
    if soup.title and soup.title.text:
        return 1
    return 0

def has_input(soup):
    if len(soup.find_all("input")) > 0:
        return 1
    return 0

def has_button(soup):
    if len(soup.find_all("button")) > 0:
        return 1
    return 0

def has_image(soup):
    if len(soup.find_all("img")) > 0:
        return 1
    return 0

def has_submit(soup):
    for button in soup.find_all("input"):
        if button.get("type") == "submit":
            return 1
    return 0

def has_link(soup):
    if len(soup.find_all("link")) > 0:
        return 1
    return 0

def has_password(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") == "password" or 
            input_tag.get("name") == "password" or 
            input_tag.get("id") == "password"):
            return 1
    return 0

def has_email_input(soup):
    for input_tag in soup.find_all("input"):
        if (input_tag.get("type") == "email" or 
            input_tag.get("name") == "email" or 
            input_tag.get("id") == "email"):
            return 1
    return 0

def has_hidden_element(soup):
    for input_tag in soup.find_all("input"):
        if input_tag.get("type") == "hidden":
            return 1
    return 0

def has_audio(soup):
    if len(soup.find_all("audio")) > 0:
        return 1
    return 0

def has_video(soup):
    if len(soup.find_all("video")) > 0:
        return 1
    return 0

# --- Quantitative Features ---
def number_of_inputs(soup):
    return len(soup.find_all("input"))

def number_of_buttons(soup):
    return len(soup.find_all("button"))

def number_of_images(soup):
    image_tags = len(soup.find_all("img"))
    count = 0
    for meta in soup.find_all("meta"):
        if meta.get("type") == "image" or meta.get("name") == "image":
            count += 1
    return image_tags + count

def number_of_option(soup):
    return len(soup.find_all("option"))

def number_of_list(soup):
    return len(soup.find_all("li"))

def number_of_th(soup):
    return len(soup.find_all("th"))

def number_of_tr(soup):
    return len(soup.find_all("tr"))

def number_of_href(soup):
    count = 0
    for link in soup.find_all("link"):
        if link.get("href"):
            count += 1
    return count

def number_of_paragraph(soup):
    return len(soup.find_all("p"))

def number_of_script(soup):
    return len(soup.find_all("script"))

def length_of_title(soup):
    if soup.title and soup.title.text:
        return len(soup.title.text.strip())
    return 0