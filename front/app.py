import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests

def load_image(image_file):
    img = Image.open(image_file)
    return img

def draw_rectangles(image_path, rectangles):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for rect_info in rectangles:
        coordinates = (rect_info["xmin"], rect_info["ymin"], rect_info["xmax"], rect_info["ymax"])
        text = rect_info["name"]

        draw.rectangle(coordinates, outline="red", width=3)
        draw.text((coordinates[0], coordinates[1] - 20), text, fill="red", font=font)

    return img

st.set_page_config(page_title="OZ Fish", page_icon=":fish:", layout="centered")

"""
# OZ FISH 🐟
"""
image_file = st.file_uploader("Please upload Image/Video")

if image_file is not None:
    file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": image_file.size}
    st.write(file_details)
    st.image(load_image(image_file))

    url = "http://127.0.0.1:8000/"
    response = requests.get(url).json()

    rectangles = [response[box] for box in response]

    if st.button("Analyse Image"):
        st.write('I was clicked 🎉')
        drawn_image = draw_rectangles(image_file, rectangles)
        st.image(drawn_image, caption="Image with Rectangles.", use_column_width=True)
    else:
        st.write('I was not clicked 😞')
