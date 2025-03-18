
import streamlit as st
from io import BytesIO
from PyPDF2 import PdfReader
from streamlit_option_menu import option_menu
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
from PIL import Image
import pytesseract as pyt
import requests
import json
import fitz  # PyMuPDF
from time import sleep
from transformers import pipeline, logging as transformers_logging
import textwrap

import google.generativeai as genai

# Set up Google Generative AI API Key
os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-pro-latest')

def extract_output(response):
    """Extracts the desired output from the GenerateContentResponse object."""
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.text.startswith("output:"):
                return part.text.split("output: ")[1].strip()
    return None  # Return None if no output found

def extract_output_from_markdown(markdown_string):
    """Extracts the text after "This code will output:" from a markdown string."""
    match = re.search(r"This code will output:(.*)", markdown_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def to_markdown(text):
    text = text.replace('.', '*')
    return textwrap.indent(text, '>', predicate=lambda _: True)

# Suppress warnings from transformers library
transformers_logging.set_verbosity_error()

# Set up summarization pipeline
def create_summarizer():
    try:
        summarizer = pipeline("summarization")
        return summarizer
    except Exception as e:
        st.error(f"Error initializing summarizer: {e}")
        return None

# Set up translation with Gemini API
def translate_text_with_gemini(text, target_language):
    api_key = 'YOUR_GEMINI_API_KEY'
    url = 'https://api.gemini.com/translate'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'text': text,
        'target_language': target_language
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get('translated_text', 'Translation failed.')
    except requests.exceptions.RequestException as e:
        st.error(f"Error translating text: {e}")
        return "Translation failed."

def extract_and_concatenate_text(pdf_file):
    reader = PdfReader(pdf_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            words = text.split()
            concatenated_text = ' '.join(words)
            all_text.append(concatenated_text)
    return '\n'.join(all_text)

def enumerate_words(text):
    words = text.split()
    enumerated_dict = {i + 1: word for i, word in enumerate(words)}
    return enumerated_dict

def find_phone_numbers(enumerated_dict):
    phone_number_index = []
    phone_number_pattern = re.compile(r'^(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}$')
    for key, value in enumerated_dict.items():
        if phone_number_pattern.search(value):
            phone_number_index.append(key)
    return phone_number_index

def find_emails(enumerated_dict):
    email_index = []
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    for key, value in enumerated_dict.items():
        if email_pattern.search(value):
            email_index.append(key)
    return email_index

def find_number_sequences(enumerated_dict):
    number_index = []
    sequence_pattern = re.compile(r'\b\d{4} \d{4} \d{4}\b')
    phone_pattern = re.compile(r'\b\d{10}\b')
    for key, value in enumerated_dict.items():
        if sequence_pattern.search(value) or phone_pattern.search(value):
            number_index.append(key)
    return number_index

def redact_words_with_numbers(enumerated_dict):
    number_pattern = re.compile(r'\d')
    redacted_indices = []
    for key, value in enumerated_dict.items():
        if number_pattern.search(value):
            redacted_indices.append(key)
    return redacted_indices

def redacted(text, indices):
    words = text.split()
    for index in indices:
        if 0 <= index - 1 < len(words):
            words[index - 1] = "[REDACTED]"
    return ' '.join(words)

def create_pdf(text_to_insert, file_name):
    doc = SimpleDocTemplate(file_name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(text_to_insert, styles["Normal"]))
    doc.build(story)

def extract_images_from_pdf(pdf_file):
    pdf_document = fitz.Document(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))
            images.append(image)
    pdf_document.close()
    return images

def ocr_image(image):
    text = pyt.image_to_string(image)
    return text

def extract_text_from_pdf_images(pdf_file):
    images = extract_images_from_pdf(pdf_file)
    texts = [ocr_image(image) for image in images]
    return " ".join(texts)

# Load question-answering pipeline
def create_qa_pipeline():
    try:
        qa_pipeline = pipeline("question-answering")
        return qa_pipeline
    except Exception as e:
        st.error(f"Error initializing QA pipeline: {e}")
        return None

# Streamlit page configuration
st.set_page_config(page_title="PDF Sensitive Information Masker", layout="wide")

# Custom CSS for additional styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .stProgress .st-bo {
        background-color: #0d6efd;
    }
    .accordion-button:not(.collapsed) {
        color: #0d6efd;
        background-color: #e7f1ff;
    }
    .typing-animation {
        width: 100%;
        white-space: nowrap;
        overflow: hidden;
        border-right: 0.15em solid black;
        animation:
            typing 2s steps(40, end),
            blink-caret 0.75s step-end infinite;
        display: block;
    }

    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: black; }
    }
    </style>
    """, unsafe_allow_html=True)

if 'selected' not in st.session_state:
    st.session_state.selected = "Home"

with st.sidebar:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("Desktop/logo.png", width=250, use_column_width=True)
    st.markdown("<h1 style='text-align: center; font-size: 150%; font-weight: bold;'> SENSITIVE INFORMATION MASKER</h1>", unsafe_allow_html=True)
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Optical Character Recognition", "Text Summarizer", "Multilingual Support", "QnA"],
        icons=["house", "camera", "book", "translate", "question-circle"], 
        default_index=0,
    )
    st.session_state.selected = selected

if st.session_state.selected == "Home":
    st.title("Home - Sensitive Information Masking")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            
            text = extract_and_concatenate_text(uploaded_file)
            progress_bar.progress(25)
            
            enumerated = enumerate_words(text)
            progress_bar.progress(50)
            
            phone_numbers = find_phone_numbers(enumerated)
            found_sequences = find_number_sequences(enumerated)
            words_and_numbers = redact_words_with_numbers(enumerated)
            emails = find_emails(enumerated)
            progress_bar.progress(75)
            
            redacted_text = redacted(text, phone_numbers + emails + found_sequences + words_and_numbers)
            
            st.write("### Extracted Text")
            st.text_area("Extracted Text", text, height=300)

            pdf_dir = "pdf_files"
            os.makedirs(pdf_dir, exist_ok=True)
            file_name = os.path.join(pdf_dir, f"redacted_{uploaded_file.name}")
            create_pdf(redacted_text, file_name)
            progress_bar.progress(100)
            
            st.success(f"File processed and redacted PDF saved as: {file_name}")

            with open(file_name, "rb") as f:
                st.download_button(
                    label="Download Redacted PDF",
                    data=f,
                    file_name=f"redacted_{uploaded_file.name}",
                    mime="application/pdf"
                )

if st.session_state.selected == "Optical Character Recognition":
    st.title("Optical Character Recognition (OCR)")

    uploaded_file = st.file_uploader("Choose a PDF file for OCR", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            
            text = extract_text_from_pdf_images(uploaded_file)
            progress_bar.progress(50)
            
            st.text_area("Extracted Text", text, height=300)
            progress_bar.progress(100)
            
            st.success("OCR completed and text extracted.")

if st.session_state.selected == "Text Summarizer":
    st.title("Text Summarizer")
    
    uploaded_file = st.file_uploader("Choose a PDF file for summarization", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            text = extract_and_concatenate_text(uploaded_file)
            
            summarizer = create_summarizer()
            if summarizer:
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
                summary_text = summary[0]['summary_text']
                st.write(summary_text)

if st.session_state.selected == "Multilingual Support":
    st.title("Multilingual Support")
    
    uploaded_file = st.file_uploader("Choose a PDF file for redaction", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            text = extract_and_concatenate_text(uploaded_file)
            
            enumerated = enumerate_words(text)
            phone_numbers = find_phone_numbers(enumerated)
            found_sequences = find_number_sequences(enumerated)
            words_and_numbers = redact_words_with_numbers(enumerated)
            emails = find_emails(enumerated)
            
            redacted_text = redacted(text, phone_numbers + emails + found_sequences + words_and_numbers)
            
            pdf_dir = "pdf_files"
            os.makedirs(pdf_dir, exist_ok=True)
            file_name = os.path.join(pdf_dir, f"redacted_{uploaded_file.name}")
            create_pdf(redacted_text, file_name)
            
            st.success(f"File processed and redacted PDF saved as: {file_name}")
            
            st.text_area("Redacted Text", redacted_text, height=300)
            
            with open(file_name, "rb") as f:
                st.download_button(
                    label="Download Redacted PDF",
                    data=f,
                    file_name=f"redacted_{uploaded_file.name}",
                    mime="application/pdf"
                )

if st.session_state.selected == "QnA":
    st.title("QnA with PDF")
    
    uploaded_file = st.file_uploader("Upload a PDF for QnA", type=["pdf"])
    if uploaded_file:
        text = extract_and_concatenate_text(uploaded_file)
        qa_pipeline = create_qa_pipeline()
        
        if qa_pipeline:
            question = st.text_input("Ask a question about the PDF content")
            if question:
                with st.spinner("Processing your question..."):
                    answer = qa_pipeline(question=question, context=text)
                    st.write("Answer:")
                    st.write(answer['answer'])
