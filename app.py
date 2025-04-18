import streamlit as st
from login import login_page  # Import the login function from login.py
from io import BytesIO
import zipfile
import requests
from pathlib import Path
from functions.KeywordExtraction import MedicalKeywordExtractor, MedicalKeywordProcessor
from PIL import Image
import os
import json
from PIL import Image  # <-- Import PIL to handle images
from io import BytesIO
from functions.Ocr import OCRProcessor, MedicalDataExtractor
# Initialize keyword extractor

extractor = MedicalKeywordExtractor()
# 🚀 Set page configuration
st.set_page_config(page_title="Care Companion", page_icon="💙", layout="wide")

# 🔐 Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_page()  
    st.stop()  

# 🎨 Sidebar Navigation
st.sidebar.title("🔹 Care Companion")
page = st.sidebar.radio("Select a Feature", ["Home 🏠", "Unstructured to Structured 🔄", "Keyword Extraction 🔍", "Text Summarization & Translation 📜🌐", "Chatbot 🤖"])

# 🏠 Home Page
if page == "Home 🏠":
    st.title("🏥 Welcome to Care Companion!")
    st.subheader("Your AI-powered assistant for medical text processing. 🚀")
    st.write("Empowering healthcare with AI-driven solutions for better insights and analysis.")

    st.markdown("### 🔥 Why Care Companion?")
    st.write("""
    ✅ Extract *meaningful information* from raw text  
    ✅ Find *important keywords* in large medical documents  
    ✅ Summarize *long reports* into easy-to-read summaries  
    ✅ Chat with AI for quick medical insights! 🤖  
    """)
    st.info("💡 Knowledge Box: AI is transforming healthcare, reducing paperwork, and enhancing diagnostics!")
    st.image("https://source.unsplash.com/800x400/?medical,AI", caption="AI in Healthcare", use_column_width=True)
    st.markdown("### 🌟 Features:")

# 📌 Feature 1: Unstructured Data to Structured Data
elif page == "Unstructured to Structured 🔄":
    st.title("📝 Convert Unstructured Data from Images to Structured Data 📊")
    st.write("Upload medical images, extract text, and structure the data for better analysis. ✨")

    # File Uploader for JPG Images
    uploaded_images = st.file_uploader("📂 Upload medical images (JPG only):", accept_multiple_files=True, type=["jpg"])

    if st.button("Convert to Structured Data"):
        if uploaded_images:
            structured_results = []
            
            for uploaded_image in uploaded_images:
                image = Image.open(uploaded_image)
                img_path = f"temp_{uploaded_image.name}"  
                image.save(img_path)  # Temporarily save the image
                
                # Extract text using OCRProcessor
                extracted_text = OCRProcessor.extract_text(img_path)

                # Extract structured data
                structured_data = MedicalDataExtractor.extract_medical_data(extracted_text)

                # Format structured output
                structured_json = json.dumps(structured_data, indent=2, ensure_ascii=False)
                
                # Display extracted text and structured data
                st.subheader(f"📄 Extracted Text from {uploaded_image.name}")
                st.text_area("OCR Extracted Text", extracted_text, height=150)
                
                st.subheader(f"📊 Structured Data from {uploaded_image.name}")
                st.json(structured_data)

                # Remove temporary file
                os.remove(img_path)

        else:
            st.warning("⚠️ Please upload at least one JPG image.")

    st.markdown("### 🌟 Benefits:")
    st.write("""
    🔹 Extract medical data automatically from prescriptions & reports  
    🔹 Structure unstructured data for better insights  
    🔹 Save time and reduce manual data entry  
    """)

    st.success("💡 Fun Fact: OCR technology can extract handwritten and printed text with high accuracy!")

# 🔍 Feature 2: Keyword Extraction
elif page == "Keyword Extraction 🔍":
    
    st.title("🔍 Find Important Keywords in Medical Text 🏥")
    st.write("Extract key medical terms and critical insights from large documents efficiently.")
    user_input = st.text_area("📄 Enter medical text:", height=150)
    if st.button("Extract Keywords"):
        if user_input.strip():
            # Extract keywords
            keywords = extractor.extract_keywords(user_input, top_n=10)
            
            # Define categories (Modify based on requirements)
            symptoms = []
            conditions = []
            body_parts = []
            other = []

            for word, score in keywords:
                if word in ["pain", "shortness of breath"]:  # Add more symptoms
                    symptoms.append((word, score))
                elif word in ["diabetes", "hypertension"]:  # Add more conditions
                    conditions.append((word, score))
                elif word in ["heart"]:  # Add more body parts
                    body_parts.append((word, score))
                else:
                    other.append((word, score))

            # Display extracted keywords systematically
            st.markdown("## 📌 Extracted Keywords:")
            st.write("------------------------------------------------")

            if symptoms:
                st.markdown("### **SYMPTOMS:**")
                for word, score in symptoms:
                    st.write(f"🔹 {word} (score: {score:.4f})")

            if conditions:
                st.markdown("### **CONDITIONS:**")
                for word, score in conditions:
                    st.write(f"🩺 {word} (score: {score:.4f})")

            if body_parts:
                st.markdown("### **BODY PARTS:**")
                for word, score in body_parts:
                    st.write(f"❤️ {word} (score: {score:.4f})")

            if other:
                st.markdown("### **OTHER:**")
                for word, score in other:
                    st.write(f"📌 {word} (score: {score:.4f})")

        else:
            st.warning("⚠️ Please enter some text to extract keywords.")
    
    st.markdown("### 🏆 How it Helps:")
    st.write("""
    ✅ Saves time in analyzing reports  
    ✅ Identifies key symptoms and conditions  
    ✅ Helps doctors focus on important data  
    """)

    st.warning("💡 Fact: NLP-based keyword extraction helps detect diseases faster!")

# 📜 Feature 2 & 3 Combined: Text Summarization & Translation
elif page == "Text Summarization & Translation 📜🌐":
    st.title("📜🌐 Medical Text Summarization & Translation")

    # Input text area for user input
    text_input = st.text_area("📝 Enter medical text to summarize:", height=200)

    if st.button("Generate Summary"):
        if text_input:
            from functions.Summarization import MedicalSummary
            summarizer = MedicalSummary()
            summary = summarizer.summarize_text(text_input)
            st.session_state.current_summary = summary
            st.success("✅ Summary generated successfully!")
            st.write("### Summary:")
            st.write(summary)
        else:
            st.error("❌ No input provided. Please enter medical text.")

# ---------------------- Translation Section ----------------------
    st.write("🌐 **Translate the generated summary into different languages.**")

# Language selection
    target_language = st.selectbox(
    "Select target language",
    ['gujarati', 'hindi', 'marathi'],
    format_func=lambda x: x.capitalize()
)

# Translate button logic
    if st.button("Translate Summary"):
        if st.session_state.get("current_summary"):
            from functions.Translation import translate_medical_summary
        
        # Call the translation function
            translation_result = translate_medical_summary(
            st.session_state.current_summary,
            target_language
        )
        
        # Handle result
        if translation_result.get("status") == "success":
            st.success("✅ Translation completed successfully!")
            st.markdown("### 📝 Original Summary:")
            st.write(translation_result['original'])
            st.markdown("### 🌐 Translated Summary:")
            st.write(translation_result['translated'])
        else:
            st.error("❌ Translation failed. Please try again.")
    else:
        st.error("❌ No summary available to translate. Please generate a summary first.")

# Tip for users
        st.info("💡 You can translate summaries into Gujarati, Hindi, or Marathi for better accessibility.")

# 🤖 Chatbot Feature
elif page == "Chatbot 🤖":
    st.title("🗣️ Care Companion Chatbot")
    st.write("💬 Ask me anything related to medical insights and reports!")

    from functions.Chatbot import Chatbot, DataProcessor

    # Initialize chatbot once
    if 'chatbot' not in st.session_state:
        data_processor = DataProcessor()
        st.session_state['chatbot'] = Chatbot(data_processor)

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # User input
    user_input = st.text_input("🧑 You:")

    if user_input:
        chatbot = st.session_state['chatbot']
        best_question, response, source = chatbot.get_response(user_input)
        formatted_response = f"{response}\n\n*(Source: {source})*" if source != "N/A" else response
        st.session_state['chat_history'].append((user_input, formatted_response))

    # Display chat history
    for user_msg, bot_response in st.session_state['chat_history']:
        st.markdown(f"**🧑 You:** {user_msg}")
        st.markdown(f"**🤖 CareBot:** {bot_response}")

    st.info("💡 Tip: The chatbot can answer medical queries, summarize reports, and suggest keywords!")

# 🎨 Footer
st.sidebar.markdown("---")
st.sidebar.write("💙 *Care Companion - AI for Healthcare!* 🚀")
