# ML Services - Flask Web Application

A comprehensive Flask-based web application featuring multiple AI and Machine Learning models for image classification, text generation, translation, sentiment analysis, and question answering.

## Features

### 1. **Image Classification** 🖼️

- **Model**: CNN (Convolutional Neural Network)
- **Task**: Binary classification (Male/Female)
- **Input**: Image file (JPG, PNG, etc.)
- **Output**: Classification label and confidence score

### 2. **Text Generation** ✨

- **Model**: GPT-2 (Hugging Face Transformers)
- **Task**: Generate text continuations
- **Input**: Text prompt
- **Output**: Generated text

### 3. **Translation** 🌐

- **Model**: Helsinki-NLP OPUS-MT
- **Task**: English to Urdu translation
- **Input**: English text
- **Output**: Urdu translation

### 4. **Sentiment Analysis** 😊

- **Model**: Transformers (RoBERTa-based)
- **Task**: Analyze sentiment from voice
- **Input**: Audio file (MP3, WAV, etc.)
- **Process**: Speech-to-Text → Sentiment Analysis
- **Output**: Transcribed text + Sentiment label + Confidence

### 5. **Question Answering** ❓

- **Model**: DistilBERT-based QA (SQuAD)
- **Task**: Answer questions with voice input/output
- **Input**: Audio file containing question
- **Process**: Speech-to-Text → QA Model → Text-to-Speech
- **Output**: Audio file with answer

## Project Structure

```
Assignment-3/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── requirements-extra.txt      # Additional dependencies for transformers & audio
├── models/
│   ├── service.py             # Model wrapper service
│   ├── cnn_image_classification.py
│   ├── sentiment_analysis.py
│   ├── question_answer.py
│   ├── text_generation.py
│   ├── translation.py
│   ├── speech_to_text.py
│   └── text_to_speech.py
├── templates/
│   ├── index.html             # Home page with service links
│   ├── image.html             # Image classification page
│   ├── text.html              # Text generation & translation
│   ├── sentiment.html         # Sentiment analysis page
│   ├── qa.html                # Question answering page
│   └── docs.html              # API documentation
├── static/
│   ├── css/
│   │   └── styles.css         # Beautiful Bootstrap-based styling
│   ├── js/
│   │   └── app.js             # Frontend JavaScript
│   └── uploads/               # Temporary file storage
└── data/                       # Training data directory
```

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**

   ```bash
   cd Assignment-3
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-extra.txt
   ```

4. **Create necessary directories**

   ```bash
   mkdir -p static/uploads
   mkdir -p models
   ```

## Running the Application

### Start the Flask server

```bash
python app.py
```

The application will be available at: **<http://localhost:5000>**

### Development mode (with auto-reload)

The app is already configured to run in debug mode, so it will automatically reload when you make changes.

## Usage

### Home Page

Navigate to <http://localhost:5000> to see all available services.

### Image Classification

1. Go to `/image`
2. Upload a male or female image
3. View classification results with confidence scores

### Text Generation

1. Go to `/text#gen`
2. Enter a prompt
3. Get generated text continuation

### Translation

1. Go to `/text#trans`
2. Enter English text
3. Get Urdu translation

### Sentiment Analysis

1. Go to `/sentiment`
2. Upload audio file
3. View transcribed text and sentiment

### Question Answering

1. Go to `/qa`
2. Upload audio with question
3. Download audio response

### API Documentation

Visit `/docs` for detailed API endpoint documentation.

## API Endpoints

### Image Classification

```
POST /api/image_classify
Content-Type: multipart/form-data

Parameters:
  - image: file (image file to classify)

Response:
{
  "label": "male",
  "score": 0.95,
  "raw_predictions": [0.05, 0.95]
}
```

### Text Generation

```
POST /api/text_generate
Content-Type: application/json

Body:
{
  "prompt": "One day Pakistan will be"
}

Response:
{
  "text": "One day Pakistan will be a nation of..."
}
```

### Translation

```
POST /api/translate
Content-Type: application/json

Body:
{
  "text": "Pakistan is a beautiful country"
}

Response:
{
  "text": "پاکستان ایک خوبصورت ملک ہے"
}
```

### Sentiment Analysis

```
POST /api/sentiment_voice
Content-Type: multipart/form-data

Parameters:
  - audio: file (audio file)

Response:
{
  "text": "I love this application",
  "sentiment": {
    "label": "POSITIVE",
    "score": 0.98
  }
}
```

### Question Answering

```
POST /api/qa_voice
Content-Type: multipart/form-data

Parameters:
  - audio: file (audio file with question)

Response:
Binary audio file (MP3) with answer
```

## Technologies Used

### Backend

- **Flask**: Web framework
- **Flask-CORS**: Cross-Origin Resource Sharing
- **Werkzeug**: WSGI utilities

### ML/AI Models

- **TensorFlow/Keras**: CNN model training and inference
- **Transformers**: GPT-2, DistilBERT, OPUS-MT models
- **PyTorch**: Underlying framework for transformers

### Audio Processing

- **speech_recognition**: Google Speech Recognition API
- **gTTS**: Google Text-to-Speech
- **pydub**: Audio format conversion
- **librosa**: Audio processing library
- **soundfile**: Audio I/O

### Frontend

- **Bootstrap 5**: Responsive CSS framework
- **Font Awesome**: Icon library
- **Vanilla JavaScript**: Frontend interactions

## Important Notes

### Model Dependencies

- **Speech Recognition**: Requires internet connection for Google Speech API
- **Text-to-Speech**: Uses Google TTS service
- **Transformers**: First run will download models (~2-5 GB)
- **CNN Model**: Train the model first and save it, or it will return stub results

### File Uploads

- Images max size: 10MB
- Audio files: MP3, WAV, M4A formats
- Uploads are temporarily stored in `static/uploads/`

### Performance

- First inference is slow (models are loaded on first request)
- Subsequent requests are faster
- Consider using production WSGI server (Gunicorn) for deployment

## Training the CNN Model

To train the gender classification model:

```python
# This should be done separately
# The training data is in data/gender/Train/
# Reference: models/cnn_image_classification.py

# After training, save the model:
model.save('models/cnn_model')
```

## Troubleshooting

### ImportError for transformers

```bash
pip install transformers torch
```

### Audio processing errors

```bash
pip install librosa soundfile pydub
```

### gTTS errors

Ensure you have internet connection for Google TTS service.

### Speech Recognition errors

- Check audio file format
- Ensure audio quality is good
- Verify internet connection

## Deployment

### Using Gunicorn (production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Heroku

1. Create `Procfile`:

   ```
   web: gunicorn app:app
   ```

2. Create `runtime.txt`:

   ```
   python-3.9.18
   ```

3. Deploy:

   ```bash
   heroku create
   git push heroku main
   ```

## Future Enhancements

- [ ] Database integration for result history
- [ ] User authentication
- [ ] Batch processing
- [ ] Model fine-tuning UI
- [ ] Advanced audio recording (no file needed)
- [ ] Results caching
- [ ] Multi-language support
- [ ] Docker containerization

## License

This project is for educational purposes.

## Contact & Support

For issues or questions, please refer to the documentation or contact the development team.

---

**Last Updated**: December 2025
**Version**: 1.0
