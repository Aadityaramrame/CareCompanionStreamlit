from flask import Flask, request, jsonify
from Main import CareCompanion

app = Flask(__name__)

# Initialize the CareCompanion main class
care_companion = CareCompanion()

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    summary = care_companion.summarize_text(text)
    return jsonify({'summary': summary})

@app.route('/Ocr', methods=['POST'])
def ocr():
    data = request.get_json()
    pdf_path = data.get('pdf_path')
    if not pdf_path:
        return jsonify({'error': 'No PDF path provided'}), 400

    extracted_text = care_companion.extract_text_from_pdf(pdf_path)
    return jsonify({'extracted_text': extracted_text})

@app.route('/KeywordExtraction', methods=['POST'])
def keyword_extraction():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    keywords = care_companion.extract_keywords(text)
    return jsonify({'keywords': keywords})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
