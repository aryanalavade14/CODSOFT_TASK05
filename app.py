import os
import sys
import json
import textwrap
from io import BytesIO

import torch
import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(BASE_DIR, "src"))

from model import HandwritingCNN

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "handwriting_cnn.pth"
)

MAPPING_PATH = os.path.join(
    BASE_DIR, "dataset", "processed", "label_mapping.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="InkFlow AI | Handwriting Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 3. HTML RENDERING
# =====================================================

def render_html(content):
    st.html(textwrap.dedent(content))


# =====================================================
# 4. PROFESSIONAL NAVY BLUE THEME
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #081426;
    color: #F4F2FF;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: #081426;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* NAVIGATION */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 25px;
    border-bottom: 1px solid #263751;
    margin-bottom: 35px;
}

.brand {
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.brand span {
    color: #B69AFF;
}

.nav-label {
    color: #B8C3D8;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}

/* HERO */

.hero {
    text-align: center;
    padding: 35px 15px 42px;
}

.eyebrow {
    display: inline-block;
    background: #1B2945;
    color: #C4A9FF;
    border: 1px solid #49366F;
    border-radius: 30px;
    padding: 8px 17px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 20px;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 47px;
    color: #FFFFFF;
    line-height: 1.25;
    margin: 0 0 15px;
}

.hero h1 span {
    color: #B69AFF;
}

.hero p {
    font-size: 16px;
    color: #B7C3D8;
    max-width: 680px;
    margin: auto;
    line-height: 1.8;
}

/* FEATURE CARDS */

.feature-card {
    background: #12213A;
    border: 1px solid #293B59;
    border-radius: 15px;
    padding: 21px;
    min-height: 120px;
}

.feature-icon {
    font-size: 22px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 12px;
    color: #AAB8CF;
    font-weight: 600;
    margin-bottom: 7px;
}

.feature-value {
    font-size: 21px;
    color: #FFFFFF;
    font-weight: 800;
}

/* SECTION HEADINGS */

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #FFFFFF;
    margin-top: 32px;
    margin-bottom: 7px;
}

.section-description {
    color: #B0BCD1;
    font-size: 14px;
    margin-bottom: 18px;
}

/* TEXT INPUT */

.stTextArea textarea {
    background: #12213A !important;
    color: #FFFFFF !important;
    border: 1px solid #435575 !important;
    border-radius: 13px !important;
    padding: 15px !important;
    font-size: 16px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextArea textarea::placeholder {
    color: #AAB8CF !important;
    opacity: 1 !important;
}

.stTextArea textarea:focus {
    border-color: #B69AFF !important;
    box-shadow: 0 0 0 2px #49366F !important;
}

.stTextArea label,
.stTextArea label p {
    color: #FFFFFF !important;
}

[data-testid="stCaptionContainer"] {
    color: #B7C3D8 !important;
}

/* GENERATE BUTTON */

.stButton > button {
    background: linear-gradient(90deg, #7044D5, #9868E8);
    color: #FFFFFF !important;
    border: none;
    border-radius: 11px;
    min-height: 49px;
    font-size: 15px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #865DE5, #AD82F2);
    color: #FFFFFF !important;
    transform: translateY(-2px);
}

/* RESULT PANEL */

.result-panel {
    background: #12213A;
    border: 1px solid #293B59;
    border-radius: 17px;
    padding: 23px;
    margin-top: 15px;
}

.result-heading {
    color: #D2BFFF;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 15px;
}

.result-note {
    color: #AAB8CF;
    font-size: 12px;
    margin-top: 12px;
}

/* DOWNLOAD BUTTON */

.stDownloadButton > button {
    background: #182A45;
    color: #D5C4FF !important;
    border: 1px solid #8061B9;
    border-radius: 11px;
    min-height: 48px;
    font-weight: 700;
}

.stDownloadButton > button:hover {
    background: #263A5B;
    border-color: #B69AFF;
    color: #FFFFFF !important;
}

/* ALERTS */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #101D33;
    border-right: 1px solid #293B59;
}

[data-testid="stSidebar"] * {
    color: #E7E9F4;
}

/* FOOTER */

.footer {
    border-top: 1px solid #293B59;
    margin-top: 50px;
    padding: 25px 0;
    text-align: center;
    color: #AAB8CF;
    font-size: 13px;
}

.footer strong {
    color: #C4A9FF;
}

/* MOBILE */

@media (max-width: 700px) {
    .hero h1 {
        font-size: 33px;
    }

    .nav-label {
        font-size: 9px;
    }

    .brand {
        font-size: 22px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 5. LOAD TRAINED CNN MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("CNN model file not found.")

    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError("Character mapping not found.")

    with open(MAPPING_PATH, "r") as file:
        mapping = json.load(file)

    model = HandwritingCNN(
        vocab_size=62,
        embedding_dim=128,
        image_size=96
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    return model, mapping


# =====================================================
# 6. GENERATE INDIVIDUAL CHARACTER
# =====================================================

def generate_character(character, model, mapping):

    character_index = mapping[character]

    input_tensor = torch.tensor(
        [[character_index]],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)

    generated_image = output.squeeze().cpu().numpy()

    return np.clip(generated_image, 0, 1)


# =====================================================
# 7. PREPARE MEDIUM-SIZED SHARP CHARACTER
# =====================================================

def prepare_character(character, model, mapping):

    generated = generate_character(
        character,
        model,
        mapping
    )

    image = Image.fromarray(
        (generated * 255).astype(np.uint8)
    )

    # Crop white margins

    array = np.array(image)
    mask = array < 245

    if np.any(mask):

        rows = np.where(np.any(mask, axis=1))[0]
        cols = np.where(np.any(mask, axis=0))[0]

        top = max(0, rows[0] - 2)
        bottom = min(96, rows[-1] + 3)

        left = max(0, cols[0] - 2)
        right = min(96, cols[-1] + 3)

        image = image.crop(
            (left, top, right, bottom)
        )

    # Medium character height

    new_height = 44

    new_width = max(
        1,
        int(image.width * new_height / image.height)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Improve contrast and sharpness

    image = ImageOps.autocontrast(
        image,
        cutoff=1
    )

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    image = image.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=120,
            threshold=3
        )
    )

    return image


# =====================================================
# 8. BUILD COMPLETE WORD IMAGE
# =====================================================

def build_word(
    word,
    model,
    mapping,
    character_spacing=8
):

    character_images = [
        prepare_character(character, model, mapping)
        for character in word
    ]

    total_width = (
        sum(image.width for image in character_images)
        + character_spacing * (len(character_images) - 1)
    )

    word_image = Image.new(
        "L",
        (max(1, total_width), 44),
        color=255
    )

    x_position = 0

    for image in character_images:

        y_position = (44 - image.height) // 2

        word_image.paste(
            image,
            (x_position, y_position)
        )

        x_position += image.width + character_spacing

    return word_image


# =====================================================
# 9. BUILD COMPLETE LINE IMAGE
# =====================================================

def build_line(
    word_images,
    word_spacing=28,
    line_height=58
):

    total_width = (
        sum(image.width for image in word_images)
        + word_spacing * (len(word_images) - 1)
    )

    line_image = Image.new(
        "L",
        (max(1, total_width), line_height),
        color=255
    )

    x_position = 0

    for image in word_images:

        y_position = (line_height - image.height) // 2

        line_image.paste(
            image,
            (x_position, y_position)
        )

        x_position += image.width + word_spacing

    return line_image


# =====================================================
# 10. GENERATE TEXT - PRESERVE INPUT LINES
# =====================================================

def generate_text(text, model, mapping):

    text = text.replace(
        "\r\n", "\n"
    ).replace(
        "\r", "\n"
    )

    input_lines = text.split("\n")

    generated_lines = []

    line_height = 58
    line_spacing = 24
    word_spacing = 28

    for input_line in input_lines:

        # Preserve empty lines and paragraph spacing

        if not input_line.strip():

            generated_lines.append(None)
            continue

        words = input_line.split()

        word_images = []

        for word in words:

            word_image = build_word(
                word,
                model,
                mapping,
                character_spacing=8
            )

            word_images.append(word_image)

        # Keep the complete input line together

        line_image = build_line(
            word_images,
            word_spacing=word_spacing,
            line_height=line_height
        )

        generated_lines.append(line_image)

    if not generated_lines or all(
        line is None for line in generated_lines
    ):

        raise ValueError("Please enter some text.")

    actual_lines = [
        line for line in generated_lines
        if line is not None
    ]

    max_width = max(
        line.width for line in actual_lines
    )

    total_height = (
        len(generated_lines) * line_height
        + (len(generated_lines) - 1) * line_spacing
    )

    # Create final white paper

    final_image = Image.new(
        "L",
        (max_width, total_height),
        color=255
    )

    y_position = 0

    for line in generated_lines:

        if line is not None:

            final_image.paste(
                line,
                (0, y_position)
            )

        y_position += line_height + line_spacing

    return final_image


# =====================================================
# 11. NAVIGATION
# =====================================================

render_html("""
<div class="navbar">

    <div class="brand">
        ✍ Ink<span>Flow</span>
    </div>

    <div class="nav-label">
        AI HANDWRITING STUDIO &nbsp; • &nbsp; DEEP LEARNING
    </div>

</div>
""")


# =====================================================
# 12. HERO SECTION
# =====================================================

render_html("""
<div class="hero">

    <div class="eyebrow">
        INTRODUCING INKFLOW AI
    </div>

    <h1>
        Every word has a <span>personal touch.</span>
    </h1>

    <p>
        Transform your typed words into handwritten-style images.
        Experience character-level text generation powered by
        a Convolutional Neural Network.
    </p>

</div>
""")


# =====================================================
# 13. FEATURE CARDS
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:

    render_html("""
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">MODEL ARCHITECTURE</div>
        <div class="feature-value">CNN Model</div>
    </div>
    """)

with col2:

    render_html("""
    <div class="feature-card">
        <div class="feature-icon">🔤</div>
        <div class="feature-title">SUPPORTED CLASSES</div>
        <div class="feature-value">62 Characters</div>
    </div>
    """)

with col3:

    render_html("""
    <div class="feature-card">
        <div class="feature-icon">🖼️</div>
        <div class="feature-title">OUTPUT RESOLUTION</div>
        <div class="feature-value">96 × 96</div>
    </div>
    """)


# =====================================================
# 14. TEXT INPUT
# =====================================================

render_html("""
<div class="section-title">
    Create your handwriting
</div>

<div class="section-description">
    Enter a name, message or multiple paragraphs.
    Each manually entered line stays on one line.
</div>
""")

user_text = st.text_area(
    "Text input",
    placeholder="Welcome to InkFlow AI!",
    height=180,
    max_chars=1000,
    label_visibility="collapsed"
)

left_column, right_column = st.columns([1, 2])

with left_column:

    generate_button = st.button(
        "✦ Generate Handwriting",
        use_container_width=True
    )

with right_column:

    st.caption(
        "Supports multiple lines, paragraphs, A-Z, a-z, 0-9 and spaces."
    )


# =====================================================
# 15. GENERATE AND DISPLAY HANDWRITING
# =====================================================

if generate_button:

    if not user_text.strip():

        st.warning(
            "Please enter some text before generating."
        )

    else:

        try:

            model, mapping = load_model()

            unsupported = sorted(set(
                character
                for character in user_text
                if character not in (" ", "\n", "\r", "\t")
                and character not in mapping
            ))

            if unsupported:

                st.error(
                    "Unsupported characters: "
                    + ", ".join(unsupported)
                )

            else:

                with st.spinner(
                    "Creating your handwriting..."
                ):

                    result = generate_text(
                        user_text,
                        model,
                        mapping
                    )

                st.success(
                    "Your handwriting is ready!"
                )

                render_html("""
                <div class="section-title">
                    Your generated handwriting
                </div>

                <div class="section-description">
                    Preview your generated text below.
                </div>

                <div class="result-panel">
                    <div class="result-heading">
                        ✍ Generated Preview
                    </div>
                </div>
                """)

                # Medium preview, without stretching

                display_width = min(
                    result.width,
                    850
                )

                st.image(
                    result,
                    width=display_width
                )

                render_html("""
                <div class="result-note">
                    Generated using your trained CNN model.
                </div>
                """)

                # Download PNG

                image_bytes = BytesIO()

                result.save(
                    image_bytes,
                    format="PNG"
                )

                st.write("")

                st.download_button(
                    label="↓ Download PNG Image",
                    data=image_bytes.getvalue(),
                    file_name="inkflow_handwriting.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Unable to generate handwriting: {error}"
            )


# =====================================================
# 16. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## ✍ InkFlow AI")

    st.caption(
        "Handwritten Text Generation"
    )

    st.divider()

    st.markdown("### About")

    st.write(
        "An AI-powered character-level handwritten text "
        "generation application built using a CNN model."
    )

    st.divider()

    st.markdown("### Technology Stack")

    st.write("""
    - Python
    - PyTorch
    - Convolutional Neural Network
    - Streamlit
    - NumPy
    - Pillow
    """)

    st.divider()

    st.markdown("### Model Configuration")

    st.write("""
    - Input: Character ID
    - Model: CNN
    - Character classes: 62
    - Original resolution: 96 × 96
    - Display character height: 44 pixels
    - Letter spacing: 8 pixels
    - Word spacing: 28 pixels
    - Line spacing: 24 pixels
    - Paragraph support: Yes
    """)


# =====================================================
# 17. FOOTER
# =====================================================

render_html("""
<div class="footer">

    <strong>InkFlow AI</strong>

    <br><br>

    Handwritten Text Generation using Deep Learning

    <br>

    Built with Python, PyTorch and Streamlit

</div>
""")