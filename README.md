# InkFlow AI – Handwritten Text Generation

InkFlow AI is a deep learning-based handwritten text generation application that transforms typed text into handwritten-style images using a Convolutional Neural Network (CNN).

The application provides an interactive interface where users can enter text, generate handwritten output, customize the appearance, and download the generated result as a PNG image.

## Features

- Generate handwritten-style text from user input.
- CNN-based character image generation.
- Support for multiple words, sentences, and paragraphs.
- Customizable character spacing and text appearance.
- Image sharpening and contrast enhancement.
- Download generated handwriting as PNG.
- Interactive web interface built using Streamlit.

## Technologies Used

- Python
- Deep Learning
- Convolutional Neural Network (CNN)
- PyTorch
- Streamlit
- NumPy
- Pillow (PIL)

## Project Structure

```text
Handwritten_Text_Generation/
│
├── app.py
├── requirements.txt
├── generated_handwriting.png
│
├── dataset/
│   ├── Img/
│   ├── english.csv
│   └── processed/
│       ├── label_mapping.json
│       └── processed_dataset.npz
│
├── models/
│   ├── handwriting_cnn.pth
│   ├── cnn_training_loss.png
│   └── training_loss.png
│
└── src/
    ├── generate.py
    ├── model.py
    ├── preprocessing.py
    └── train.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aryanalavade14/CODSOFT_TASK05.git
```

### 2. Navigate to the project directory

```bash
cd CODSOFT_TASK05
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How It Works

1. The user enters text into the application.
2. The input is processed character by character.
3. The trained CNN model generates handwritten-style character images.
4. Generated characters are combined to form words and sentences.
5. Image enhancement techniques improve the final output.
6. The generated handwriting can be downloaded as a PNG image.

## Future Improvements

- More diverse handwriting styles.
- Improved character-level consistency.
- Support for additional languages.
- Sequence-based handwriting generation.
- Personalized handwriting generation.

## Author

**Arya Nalavade**

Electronics & Computer Science Engineering

GitHub: [@aryanalavade14](https://github.com/aryanalavade14)

---

*InkFlow AI – Turning Digital Text into Handwritten Art.*
