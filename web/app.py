import sys
import os
import pickle
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Importar configuración central ───────────────────────────
# Para cambiar dataset o categorías: edita src/config.py
from config import CATEGORY_DISPLAY, CATEGORY_COLOR, CATEGORIES

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'naive_bayes_model.pkl')

print("Cargando modelo Naïve Bayes...")
with open(MODEL_PATH, 'rb') as f:
    MODEL_DATA = pickle.load(f)
print("✓ Modelo cargado correctamente")

CLASSES = MODEL_DATA.get('classes', [])
print(f"✓ {len(CLASSES)} categorías: {CLASSES}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Clasifica un ticket de soporte.
    Recibe:  { "text": "...", "subject": "..." }
    Retorna: predicción, display name, color y probabilidades.
    """
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Se requiere el campo "text"'}), 400

    text    = data.get('text', '').strip()
    subject = data.get('subject', '').strip()

    if not text:
        return jsonify({'error': 'El texto no puede estar vacío'}), 400

    full_text = f"{subject} {text}".strip()

    preprocessor    = MODEL_DATA['preprocessor']
    tokens          = preprocessor.preprocess(full_text)

    if not tokens:
        return jsonify({'error': 'El texto no contiene palabras válidas tras el preprocesamiento'}), 400

    classifier      = MODEL_DATA['classifier']
    predicted_class = classifier.predict_one(tokens)
    probabilities   = classifier.predict_proba(tokens)

    response = {
        'predicted_class': predicted_class,
        'display_name':    CATEGORY_DISPLAY.get(predicted_class, predicted_class),
        'color':           CATEGORY_COLOR.get(predicted_class, '#888888'),
        'probabilities': {
            CATEGORY_DISPLAY.get(k, k): v
            for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        },
        'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tokens_used': len(tokens),
    }

    return jsonify(response)


@app.route('/model-stats')
def model_stats():
    """Estadísticas del modelo para la vista AI Model."""
    eval_data  = MODEL_DATA.get('eval', {})
    classifier = MODEL_DATA['classifier']

    if not eval_data:
        return jsonify({
            'has_eval':         False,
            'classes':          CLASSES,
            'vocab_size':       getattr(classifier, 'vocab_size', 0),
            'category_display': CATEGORY_DISPLAY,
            'category_color':   CATEGORY_COLOR,
        })

    return jsonify({
        'has_eval':         True,
        'classes':          CLASSES,
        'k_folds':          eval_data.get('k', 5),
        'accuracy':         eval_data.get('accuracy', 0),
        'macro_f1':         eval_data.get('macro_f1', 0),
        'n_docs':           eval_data.get('n_docs', 0),
        'vocab_size':       eval_data.get('vocab_size', getattr(classifier, 'vocab_size', 0)),
        'per_class':        eval_data.get('per_class', {}),
        'docs_per_class':   eval_data.get('docs_per_class', {}),
        'category_display': CATEGORY_DISPLAY,
        'category_color':   CATEGORY_COLOR,
    })


@app.route('/health')
def health():
    return jsonify({
        'status':        'ok',
        'model_loaded':  MODEL_DATA is not None,
        'categories':    CLASSES,
        'total_classes': len(CLASSES),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
