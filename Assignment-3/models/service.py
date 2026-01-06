import os
from pathlib import Path
import uuid
import wave
import struct
from networkx import clustering
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN , KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Disable TensorFlow entirely - use PyTorch only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'
# Prevent transformers from importing TensorFlow
try:
    os.environ['USE_TF'] = '0'
    os.environ['USE_TORCH'] = '1'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Disable tokenizer parallelism to avoid deadlocks/warnings in multiprocess environments
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
except:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / 'static' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global cache for lazy-loaded models (all heavy imports deferred)
_sentiment_pipeline = None
_qa_pipeline = None
_text_gen_pipeline = None
_translation_pipeline = None
_cnn_model = None

img_size = (128, 128)



def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            # Disable TensorFlow and use PyTorch
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            from transformers import pipeline
            _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except Exception as e:
            print(f"Sentiment pipeline error: {e}")
    return _sentiment_pipeline


def _get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from transformers import pipeline
            _qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            print(f"QA pipeline error: {e}")
    return _qa_pipeline


def _get_text_gen_pipeline():
    global _text_gen_pipeline
    if _text_gen_pipeline is None:
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            from transformers import pipeline
            _text_gen_pipeline = pipeline("text-generation", model="distilgpt2")
        except Exception as e:
            print(f"Text generation pipeline error: {e}")
    return _text_gen_pipeline


def _get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        try:
            # Try to load without causing TensorFlow import
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from transformers import pipeline
                _translation_pipeline = pipeline("translation_en_to_ur", model="Helsinki-NLP/opus-mt-en-ur")
        except Exception as e:
            print(f"Translation pipeline error: {e}")
            print("Translation will use stub responses")
    return _translation_pipeline


def get_cnn_model():
    """Load CNN model if available"""
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    try:
        import pickle
        # Try .pkl first, then .keras, then .h5
        model_path_pkl = BASE_DIR / 'models' / 'cnn_model.pkl'
        model_path_keras = BASE_DIR / 'models' / 'cnn_model.keras'
        model_path_h5 = BASE_DIR / 'models' / 'cnn_model.h5'

        if model_path_pkl.exists():
            with open(str(model_path_pkl), 'rb') as f:
                _cnn_model = pickle.load(f)
            print(f"CNN model loaded from {model_path_pkl}")
        elif model_path_keras.exists():
            import tensorflow as tf
            _cnn_model = tf.keras.models.load_model(str(model_path_keras))
            print(f"CNN model loaded from {model_path_keras}")
        elif model_path_h5.exists():
            import tensorflow as tf
            _cnn_model = tf.keras.models.load_model(str(model_path_h5))
            print(f"CNN model loaded from {model_path_h5}")
        else:
            print(f"CNN model not found at {model_path_pkl}, {model_path_keras}, or {model_path_h5}")
    except ImportError as e:
        print(f"Import error loading CNN model: {e}")
    except Exception as e:
        print(f"CNN model load error: {e}")
    return _cnn_model


def classify_image(image_path: str):
    """Classify gender from image using CNN"""
    try:
        model = get_cnn_model()
        if model is None:
            return {'label': 'unknown', 'score': 0.0, 'note': 'Model not loaded. Train and save model first.'}

        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        arr = np.array(img) / 255.0
        arr = arr.reshape((1, img_size[0], img_size[1], 3))
        preds = model.predict(arr)
        label = int(preds[0].argmax())
        score = float(preds[0].max())
        return {
            'label': 'male' if label == 1 else 'female',
            'score': score,
            'raw_predictions': [float(p) for p in preds[0]]
        }
    except Exception as e:
        return {'error': str(e), 'label': 'unknown'}


def generate_text(prompt: str):
    """Generate text using GPT-2"""
    try:
        pipeline = _get_text_gen_pipeline()
        if pipeline is None:
            return prompt + ' (model not loaded)'
        result = pipeline(prompt, max_length=50, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"


def translate_en_to_ur(text: str):
    """Translate English to Urdu"""
    try:
        pipeline = _get_translation_pipeline()
        if pipeline is None:
            return text + ' (model not loaded)'
        result = pipeline(text)
        return result[0]['translation_text']
    except Exception as e:
        return f"Error: {str(e)}"


def speech_to_text(audio_path: str):
    """Convert speech to text using Google Speech Recognition"""
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        # Convert to WAV if needed (for WebM, MP3, etc.)
        if not audio_path.lower().endswith('.wav'):
            try:
                from pydub import AudioSegment
                # Allow pydub to auto-detect format (works for mp3, webm, ogg, etc. if ffmpeg is installed)
                audio = AudioSegment.from_file(audio_path)
                wav_path = str(UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.wav")
                audio.export(wav_path, format="wav")
                audio_path = wav_path
            except Exception as e:
                print(f"Audio conversion warning: {e}")
                # Fallthrough to try reading as is, just in case

        # Recognize speech
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def sentiment_from_text(text: str):
    """Analyze sentiment from text"""
    try:
        pipeline = _get_sentiment_pipeline()
        if pipeline is None:
            return {'label': 'NEUTRAL', 'score': 0.0, 'note': 'Model not loaded'}
        result = pipeline(text)
        return {
            'label': result[0]['label'],
            'score': result[0]['score']
        }
    except Exception as e:
        return {'error': str(e)}


def answer_question(question: str, context: str = None):
    """Answer question using QA model"""
    try:
        pipeline = _get_qa_pipeline()
        if pipeline is None:
            return "Model not loaded"

        # Use default context if not provided
        if context is None:
            context = (
                "Pakistan is a beautiful country in South Asia. "
                "Quaid-e-Azam, or Great Leader, is a title bestowed upon Muhammad Ali Jinnah, "
                "the founding father of Pakistan. He was born on December 25, 1876, in Karachi. "
                "Jinnah's unwavering determination, political acumen, and staunch advocacy for a separate homeland "
                "for Muslims in the subcontinent led to the creation of Pakistan on August 14, 1947."
            )

        result = pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"


def text_to_speech(text: str):
    """Convert text to speech using gTTS"""
    try:
        from gtts import gTTS

        if not text or len(text.strip()) == 0:
            print("Warning: Empty text for TTS")
            text = "No response generated"

        print(f"Generating speech for: {text[:50]}...")
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(str(out_path))
        print(f"TTS saved to {out_path}")
        return str(out_path)
    except ImportError as e:
        print(f"gTTS not installed: {e}")
        # Create fallback silent WAV
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.wav"
        framerate = 16000
        duration_seconds = 1
        nframes = int(framerate * duration_seconds)
        with wave.open(str(out_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for _ in range(nframes):
                wf.writeframes(struct.pack('<h', 0))
        return str(out_path)
    except Exception as e:
        print(f"TTS error: {e}")
        # Create fallback silent WAV
        out_path = UPLOAD_DIR / f"tts_{uuid.uuid4().hex}.wav"
        framerate = 16000
        duration_seconds = 1
        nframes = int(framerate * duration_seconds)
        with wave.open(str(out_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            for _ in range(nframes):
                wf.writeframes(struct.pack('<h', 0))
        return str(out_path)

# APRIORY Algorithm
def association_rules_mining(dataSet, min_support=0.5, min_confidence=0.7):
    """
    Apply Apriori algorithm to find association rules

    Args:
        dataSet: List of transactions, each transaction is a list of items
        min_support: Minimum support threshold (0-1)
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        List of association rules as dictionaries
    """
    try:
        # Create a DataFrame for one-hot encoding
        items = sorted({item for transaction in dataSet for item in transaction})
        encoded_data = pd.DataFrame([{item: (item in transaction) for item in items} for transaction in dataSet])

        # Applying Apriori Algorithm
        frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)

        if len(frequent_itemsets) == 0:
            return []

        # Generating Association Rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        # Convert to dictionary with readable format
        result = []
        for idx, rule in rules.iterrows():
            result.append({
                'antecedents': list(rule['antecedents']),
                'consequents': list(rule['consequents']),
                'support': float(rule['support']),
                'confidence': float(rule['confidence']),
                'lift': float(rule['lift'])
            })

        return result
    except Exception as e:
        print(f"Apriori error: {e}")
        return []




# DBSCAN Clustering
def dbscan_clustering(dataSet, epsilon=10, min_samples=5):
    """
    Apply DBSCAN clustering algorithm

    Args:
        dataSet: List of numerical feature vectors
        epsilon: Maximum distance between points
        min_samples: Minimum number of samples to form a cluster

    Returns:
        List of cluster assignments for each data point
    """
    try:
        data = np.array(dataSet)

        # Applying DBSCAN
        clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
        clusters = clustering.fit_predict(data)

        return clusters.tolist()
    except Exception as e:
        print(f"DBSCAN error: {e}")
        return []



# K-Means Clustering
def kmeans_clustering(dataSet, n_clusters=2, max_iter=300, headers=None):
    """
    Apply K-Means clustering algorithm

    Args:
        dataSet: List of numerical feature vectors or DataFrame
        n_clusters: Number of clusters to create
        max_iter: Maximum number of iterations
        headers: Column names for the data

    Returns:
        Dictionary with clusters, centers, and inertia
    """
    try:
        # Convert to numpy array if list
        if isinstance(dataSet, list):
            data = np.array(dataSet)
        else:
            data = dataSet.values if hasattr(dataSet, 'values') else np.array(dataSet)

        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        clusters = kmeans.fit_predict(data)

        # Get cluster centers
        centers = kmeans.cluster_centers_.tolist()

        # Calculate inertia (sum of squared distances)
        inertia = float(kmeans.inertia_)

        return {
            'clusters': clusters.tolist(),
            'cluster_centers': centers,
            'inertia': inertia,
            'n_clusters': n_clusters,
            'dataset_raw': data.tolist()  # Include dataset for frontend plotting
        }
    except Exception as e:
        print(f"K-Means error: {e}")
        return {
            'clusters': [],
            'cluster_centers': [],
            'inertia': 0,
            'error': str(e)
        }
