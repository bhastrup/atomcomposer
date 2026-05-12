import os
from streamlit_extras.app_logo import add_logo


def show_logo():

    image_name = "atomcomposer.png"

    file_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(file_dir, image_name)

    add_logo(logo_url=image_path, height=230)
