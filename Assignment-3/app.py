from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
from models import service

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image')
def image_page():
    return render_template('image.html')


@app.route('/text')
def text_page():
    return render_template('text.html')


@app.route('/sentiment')
def sentiment_page():
    return render_template('sentiment.html')


@app.route('/qa')
def qa_page():
    return render_template('qa.html')


@app.route('/api/image_classify', methods=['POST'])
def api_image_classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)
        result = service.classify_image(str(path))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/text_generate', methods=['POST'])
def api_text_generate():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        result = service.generate_text(prompt)
        return jsonify({'text': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def api_translate():
    try:
        data = request.json or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = service.translate_en_to_ur(text)
        return jsonify({'text': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment_voice', methods=['POST'])
def api_sentiment_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        f = request.files['audio']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)
        text = service.speech_to_text(str(path))
        sentiment = service.sentiment_from_text(text)
        return jsonify({'text': text, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/qa_voice', methods=['POST'])
def api_qa_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        f = request.files['audio']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Get optional context from form data
        context = request.form.get('context')
        if context and context.strip() == "":
            context = None

        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)

        question = service.speech_to_text(str(path))
        answer = service.answer_question(question, context=context)
        audio_path = service.text_to_speech(answer)
        return send_file(audio_path, as_attachment=True, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
