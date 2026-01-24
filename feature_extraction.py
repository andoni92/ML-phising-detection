from bs4 import BeautifulSoup
import features 

def create_soup(text):
    return BeautifulSoup(text, "html.parser")

def create_vector(soup):
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