import requests
from pathlib import Path
from io import BytesIO
from PIL import Image

def get_url_from_image_path(image_path):
    base = Path('../data/h-and-m-personalized-fashion-recommendations')
    relative_path = Path(image_path).relative_to(base)
    base_url = "https://storage.googleapis.com/catalogue_hm/"
    return base_url + str(relative_path).replace("%5C", "/").replace("%5c", "/").replace('\\', '/')

def reshape():
    pass

def resize(pil_img):
    new_width = 100
    w_percent = (new_width / float(pil_img.size[0]))
    new_height = int((float(pil_img.size[1]) * float(w_percent)))
    resized_img = pil_img.resize((new_width, new_height))
    return resized_img

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    resized_img = resize(image)
    return resized_img