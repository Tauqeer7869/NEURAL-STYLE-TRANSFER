import streamlit as st
from style_transfer import run_style_transfer
from PIL import Image
import os

st.set_page_config(page_title="Neural Style Transfer", layout="centered")
st.title("🎨 Neural Style Transfer")
st.markdown("Apply **artistic styles** to your photos easily!")

# File uploaders
content_file = st.file_uploader("Upload a content image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload a style image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    content_path = f"images/{content_file.name}"
    style_path = f"images/{style_file.name}"

    with open(content_path, "wb") as f:
        f.write(content_file.read())

    with open(style_path, "wb") as f:
        f.write(style_file.read())

    st.image([content_path, style_path], caption=["Content Image", "Style Image"], width=250)

    if st.button("Generate Styled Image"):
        output_img = run_style_transfer(content_path, style_path)
        output_path = f"output/styled_{content_file.name}"
        output_img.save(output_path)

        st.image(output_path, caption="🎨 Styled Output", use_column_width=True)
        st.success("Style Transfer Complete!")
        with open(output_path, "rb") as file:
            btn = st.download_button("Download Image", data=file, file_name="styled_image.jpg", mime="image/jpeg")
