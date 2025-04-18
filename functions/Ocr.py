import cv2
import pytesseract
import torch
import numpy as np # type: ignore
from PIL import Image
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import json
import os
import glob

# --- Enhanced Medical Regex Patterns ---
MEDICAL_PATTERNS = {
    'patient': {
        'age_gender_pattern1': r'PATIENT\s*\(\s*(?P<gender1>M|F|Male|Female)\s*\)\s*/\s*(?P<age1>\d{1,3})(?=Y\b)',
        'age_gender_pattern2': r',\s*(?P<age2>\d{1,3})\s*/\s*(?P<gender2>M|F|Male|Female)\b'
    },
    'patient_extra': {
        'weight': r'Weight\s*\(Kg\)\s*:\s*(\d+)',
        'health_card': r'Health\s*Card[:\s]*Exp[:\s]*(\d{4}[\/\-]\d{2}[\/\-]\d{2})'
    },
    'clinical': {
        'diagnosis': r'(?i)Diagnosis[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine Name)',
        'vitals': {
            'bp': r'(?i)(?:BP|Blood\s*Pressure)[\s:]*(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?',
            'pulse': r'(?i)(?:Pulse|Heart\s*Rate)[\s:]*(\d{2,3})\s*(?:bpm)?',
            'temp': r'(?i)(?:Temp|Temperature)[\s:]*(\d{2}\.?\d*)\s*°?[CF]?',
            'rr': r'(?i)(?:RR|Respiratory\s*Rate)[\s:]*(\d{2})\s*(?:/min)?',
            'spo2': r'(?i)(?:SpO2|Oxygen\s*Saturation)[\s:]*(\d{2,3})\s*%?'
        },
        'complaints': r'(?i)Chief\s*Complaints[:\s-]+([\s\S]+?)(?=\n)',
        'reactions': r'(?i)(?:Adverse\s*Reactions)[\s:]+([\s\S]+?)(?=\n)',
        'investigations': r'(?i)(?:Investigations|Tests)[:\s-]+([\s\S]+?)(?=\n\s*\n|Medicine|Advice|$)'
    },
    'medications': {
        'pattern': r'(?m)^\s*\d+\)\s*((?:(?!^\s*\d+\)).)+)'
    },
    'advice': r'(?i)Advice[:\s-]+([\s\S]+?)(?=\n\s*(?:Follow\s*Up|Next\s*Visit)|$)',
    'follow_up': r'(?i)Follow\s*Up[:\s-]+(\d{2}[\/\-]\d{2}[\/\-]\d{2,4})'
}

class OCRProcessor:
    @staticmethod
    def extract_text(img_path):
        # Set the Tesseract executable path
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text
    
class MedicalDataExtractor:
    @staticmethod
    def extract_age_gender(text):
        m1 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern1'], text)
        if m1:
            return m1.group('age1').strip(), m1.group('gender1').strip()
        m2 = re.search(MEDICAL_PATTERNS['patient']['age_gender_pattern2'], text)
        if m2:
            return m2.group('age2').strip(), m2.group('gender2').strip()
        return None, None

    @staticmethod
    def extract_vitals(text):
        vitals = {}
        for vital, pattern in MEDICAL_PATTERNS['clinical']['vitals'].items():
            match = re.search(pattern, text)
            if match:
                value = match.group(1).strip().replace(' ', '')
                vitals[vital] = value
        return vitals

    @staticmethod
    def extract_medical_data(text):
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        result = {"patient": {}, "vitals": {}, "diagnosis": [], "medications": [], "investigations": [], "advice": [], "follow_up": {}}
        try:
            age, gender = MedicalDataExtractor.extract_age_gender(text)
            if age and gender:
                result["patient"].update({"age": age, "gender": gender})

            weight = re.search(MEDICAL_PATTERNS['patient_extra']['weight'], text, re.I)
            if weight:
                result["patient"]["weight"] = f"{weight.group(1).strip()} kg"

            result["vitals"] = MedicalDataExtractor.extract_vitals(text)
        
            diagnosis = re.search(MEDICAL_PATTERNS['clinical']['diagnosis'], text)
            if diagnosis:
                result["diagnosis"] = [line.strip() for line in diagnosis.group(1).split('\n') if line.strip()]

            meds = re.findall(MEDICAL_PATTERNS['medications']['pattern'], text, re.DOTALL | re.MULTILINE)
            if meds:
                result["medications"].extend([re.sub(r'\s+', ' ', m).strip() for m in meds if m.strip()])
        except Exception as e:
            print(f"Extraction error: {str(e)}")
        return result


class MedicalOCRApp:
    def process_single_image(self, img_path):
        """
        Process a single image and extract structured medical data.
        """
        processor = OCRProcessor()
        extractor = MedicalDataExtractor()

        # Extract text from the image
        extracted_text = processor.extract_text(img_path)

        # Extract structured medical data from the text
        structured_data = extractor.extract_medical_data(extracted_text)

        return structured_data

    def run(self):
        input_folder = input("Enter the folder path containing medical images: ").strip()
        if not os.path.isdir(input_folder):
            print("Invalid path. Please provide a valid folder path.")
            return
        
        processor = OCRProcessor()
        extractor = MedicalDataExtractor()

        for idx, img_path in enumerate(glob.glob(os.path.join(input_folder, '*')), 1):
            print(f"Processing Image {idx}/{len(glob.glob(os.path.join(input_folder, '*')))}: {os.path.basename(img_path)}")
            try:
                extracted_text = processor.extract_text(img_path)
                print("### Raw OCR Text:")
                print(extracted_text)

                structured_data = extractor.extract_medical_data(extracted_text)
                print("### Structured Medical Data:")
                print(json.dumps(structured_data, indent=2, ensure_ascii=False))
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {os.path.basename(img_path)}: {e}")
                print("-" * 50)
        print("Processing complete!")
if __name__ == "__main__":
    app = MedicalOCRApp()
    app.run()
