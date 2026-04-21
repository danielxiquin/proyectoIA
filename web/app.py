import sys
import os
import pickle
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'naive_bayes_model.pkl')

print("Cargando modelo Naïve Bayes...")
with open(MODEL_PATH, 'rb') as f:
    MODEL_DATA = pickle.load(f)
print("✓ Modelo cargado correctamente")

# ── Configuración dinámica de categorías ──────────────────────────────────────
# Se construye automáticamente desde las clases del modelo.
# Si cambias el dataset y reentrenan, NADA aquí necesita modificarse.

# Emojis y nombres para categorías conocidas — se usa como fallback
_KNOWN_DISPLAY = {
    "ORDER":                   "📦 Órdenes",
    "BILLING":                 "💳 Facturación",
    "SHIPPING":                "🚚 Envíos",
    "REFUND":                  "💰 Reembolsos",
    "ACCOUNT":                 "👤 Cuenta",
    "CANCEL":                  "❌ Cancelación",
    "CANCELLATION_REQUEST":    "❌ Cancelación",
    "CONTACT":                 "📞 Contacto",
    "DELIVERY":                "📬 Entrega",
    "FEEDBACK":                "⭐ Feedback",
    "NEWSLETTER_SUBSCRIPTION": "📧 Newsletter",
    "SUBSCRIPTION":            "🔄 Suscripción",
    "PAYMENT":                 "💵 Pagos",
    "INVOICE":                 "🧾 Factura",
    "TECHNICAL":               "🔧 Soporte Técnico",
    "COMPLAINT":               "⚠️ Queja",
}

# Paleta de colores — se asigna por orden a categorías nuevas
_COLOR_PALETTE = [
    "#3b82f6", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
    "#14b8a6", "#a855f7", "#fb923c", "#22d3ee", "#4ade80",
]

def _build_category_config(classes):
    """
    Genera display names y colores para cualquier lista de clases.
    - Si la clase está en _KNOWN_DISPLAY, usa el nombre/emoji configurado.
    - Si es desconocida, genera un nombre capitalizado con un emoji genérico 🏷️
    - Los colores se asignan por orden desde la paleta.
    """
    display = {}
    colors  = {}
    for i, cls in enumerate(sorted(classes)):
        display[cls] = _KNOWN_DISPLAY.get(cls, f"🏷️ {cls.replace('_', ' ').title()}")
        colors[cls]  = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
    return display, colors

# Construir config desde las clases reales del modelo cargado
CLASSES = MODEL_DATA.get('classes', [])
CATEGORY_DISPLAY, CATEGORY_COLOR = _build_category_config(CLASSES)

print(f"✓ {len(CLASSES)} categorías cargadas: {CLASSES}")


@app.route('/')
def index():
    """Sirve la página principal."""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    """
    Endpoint principal de clasificación.
    Recibe JSON: {"text": "...", "subject": "..."}
    Retorna JSON con la predicción y probabilidades.
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
        return jsonify({'error': 'El texto no contiene palabras válidas después del preprocesamiento'}), 400

    classifier      = MODEL_DATA['classifier']
    predicted_class = classifier.predict_one(tokens)
    probabilities   = classifier.predict_proba(tokens)

    ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"

    response = {
        'ticket_id':       ticket_id,
        'predicted_class': predicted_class,
        'display_name':    CATEGORY_DISPLAY.get(predicted_class, predicted_class),
        'color':           CATEGORY_COLOR.get(predicted_class, '#6b7280'),
        'probabilities': {
            CATEGORY_DISPLAY.get(k, k): v
            for k, v in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        },
        'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tokens_used': len(tokens)
    }

    return jsonify(response)


@app.route('/model-stats')
def model_stats():
    """Devuelve estadísticas del modelo Naïve Bayes para la vista de IA."""
    eval_data = MODEL_DATA.get('eval', {})
    classifier = MODEL_DATA['classifier']

    # Si no hay métricas guardadas (modelo viejo), devolver info básica del clasificador
    if not eval_data:
        import math
        n_docs_approx = None
        return jsonify({
            'has_eval':    False,
            'classes':     CLASSES,
            'vocab_size':  getattr(classifier, 'vocab_size', len(classifier.vocabulary)),
            'n_docs':      None,
            'accuracy':    None,
            'macro_f1':    None,
            'per_class':   {},
            'docs_per_class': {},
            'category_display': CATEGORY_DISPLAY,
        })

    return jsonify({
        'has_eval':      True,
        'classes':       CLASSES,
        'vocab_size':    eval_data.get('vocab_size', getattr(classifier, 'vocab_size', 0)),
        'n_docs':        eval_data.get('n_docs', 0),
        'k_folds':       eval_data.get('k', 5),
        'accuracy':      eval_data.get('accuracy', 0),
        'macro_f1':      eval_data.get('macro_f1', 0),
        'per_class':     eval_data.get('per_class', {}),
        'docs_per_class': eval_data.get('docs_per_class', {}),
        'category_display': CATEGORY_DISPLAY,
        'category_color':   CATEGORY_COLOR,
    })


@app.route('/health')
def health():
    """Endpoint para verificar que el servidor está activo."""
    return jsonify({
        'status':       'ok',
        'model_loaded': MODEL_DATA is not None,
        'categories':   CLASSES,
        'total_classes': len(CLASSES)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
