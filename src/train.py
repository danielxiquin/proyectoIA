import csv
import random
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessor import Preprocessor
from bag_of_words import BagOfWords
from naive_bayes import NaiveBayesClassifier
from evaluator import KFoldsEvaluator

# ── Configuración central ──────────────────────────────────────────────────────
# Para cambiar dataset o categorías: edita config.py
from config import DATASET_FILE, COL_TEXT, COL_LABEL


# ── 1. Cargar dataset ──────────────────────────────────────────────────────────
def load_dataset(filepath):
    X, y = [], []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text  = row.get(COL_TEXT,  '').strip()
            label = row.get(COL_LABEL, '').strip()
            if text and label:
                X.append(text)
                y.append(label)
    return X, y


# ── 2. Seed data por categoria ────────────────────────────────────────────────
# El dataset de Kaggle tiene descripciones de plantilla identicas en todas las
# categorias, lo que hace imposible aprender señal. Se agregan ejemplos claros
# por categoria (augmentacion de datos) para inicializar el vocabulario correcto.
SEED_DATA = [
    # Technical issue
    ("My laptop won't turn on after the latest software update",          "Technical issue"),
    ("The internet connection keeps dropping every few minutes",          "Technical issue"),
    ("I get a critical error message every time I open the application",  "Technical issue"),
    ("My computer crashed and I lost all unsaved work",                   "Technical issue"),
    ("The device is not recognized when I plug it into the USB port",     "Technical issue"),
    ("Network connectivity is completely down since yesterday",           "Technical issue"),
    ("The software installation fails with error code 0x80004005",        "Technical issue"),
    ("My screen flickers and goes black randomly during use",             "Technical issue"),
    ("The system freezes and becomes unresponsive every hour",            "Technical issue"),
    ("Cannot connect to wifi after the router firmware update",           "Technical issue"),
    ("The printer driver is not compatible with my operating system",     "Technical issue"),
    ("My keyboard and mouse stopped working after the Windows update",    "Technical issue"),
    ("The application crashes immediately after I log in",                "Technical issue"),
    ("My hard drive is making strange clicking noises and is very slow",  "Technical issue"),
    ("I need technical support to fix the broken configuration settings", "Technical issue"),
    ("The camera does not work and shows a black screen in video calls",  "Technical issue"),
    ("System boot takes over 10 minutes and shows hardware errors",       "Technical issue"),
    ("Cannot install the update because of a compatibility error",        "Technical issue"),
    ("The touchscreen is not responding to any touch input",              "Technical issue"),
    ("My device overheats and shuts down after a few minutes of use",     "Technical issue"),
    ("Getting SSL certificate error when trying to access the portal",    "Technical issue"),
    ("The API returns a 500 internal server error on every request",      "Technical issue"),
    ("Software bug causes the dashboard to display incorrect data",       "Technical issue"),
    ("The mobile app crashes on startup after the latest update",         "Technical issue"),
    ("Cannot synchronize data because the server connection times out",   "Technical issue"),

    # Billing inquiry
    ("I was charged twice for the same subscription this month",          "Billing inquiry"),
    ("My invoice shows an incorrect amount that does not match my plan",  "Billing inquiry"),
    ("I need a copy of my billing statement for the last three months",   "Billing inquiry"),
    ("There is an unauthorized charge on my account from last week",      "Billing inquiry"),
    ("My payment failed but the money was still deducted from my bank",   "Billing inquiry"),
    ("I did not receive a receipt for my last payment",                   "Billing inquiry"),
    ("The subscription fee was higher than the price I was quoted",       "Billing inquiry"),
    ("I want to update my credit card information for future payments",   "Billing inquiry"),
    ("I need clarification on the billing cycle and payment due date",    "Billing inquiry"),
    ("I was overcharged for the premium plan compared to the agreed fee", "Billing inquiry"),
    ("My account shows a past due balance but I paid on time",            "Billing inquiry"),
    ("I need an official invoice with VAT for my company records",        "Billing inquiry"),
    ("The transaction history shows a duplicate payment last Tuesday",    "Billing inquiry"),
    ("I cancelled my subscription but was still charged this month",      "Billing inquiry"),
    ("My bank shows the payment went through but account is still locked","Billing inquiry"),
    ("I want to switch from monthly to annual billing to save money",     "Billing inquiry"),
    ("There is a discrepancy between my invoice and what I owe",          "Billing inquiry"),
    ("I never authorized this recurring charge on my credit card",        "Billing inquiry"),
    ("Why was I billed for a service I did not use or request",           "Billing inquiry"),
    ("I need to dispute a charge because it was not what I agreed to",    "Billing inquiry"),
    ("My coupon code was not applied and I was charged full price",       "Billing inquiry"),
    ("I want to know when my next billing date is and the amount due",    "Billing inquiry"),
    ("The price on my bill does not match the promotional rate offered",  "Billing inquiry"),
    ("How do I get a refund for an accidental duplicate subscription",    "Billing inquiry"),
    ("I see a charge from your company but I never signed up",            "Billing inquiry"),

    # Product inquiry
    ("What features are included in the premium plan subscription",       "Product inquiry"),
    ("Can you explain the difference between the basic and pro plans",    "Product inquiry"),
    ("I would like more information about how the product works",         "Product inquiry"),
    ("What are the technical specifications and system requirements",     "Product inquiry"),
    ("Does the software support integration with third party tools",      "Product inquiry"),
    ("How many users can access the account simultaneously",              "Product inquiry"),
    ("What is included in the free trial and how long does it last",      "Product inquiry"),
    ("I want to compare the features of the different available plans",   "Product inquiry"),
    ("Does your service offer an API for custom integrations",            "Product inquiry"),
    ("What storage capacity is included in the standard subscription",    "Product inquiry"),
    ("Can I upgrade my current plan and keep my existing data",           "Product inquiry"),
    ("What are the payment options available for the annual plan",        "Product inquiry"),
    ("Is the product compatible with both Windows and Mac operating systems","Product inquiry"),
    ("How does the automatic backup feature work and where is data stored","Product inquiry"),
    ("What security certifications does your platform comply with",       "Product inquiry"),
    ("I need details about the enterprise plan for a large organization", "Product inquiry"),
    ("Does the product include customer support and what are the hours",  "Product inquiry"),
    ("What is the maximum file size I can upload to the platform",        "Product inquiry"),
    ("How do I export my data if I decide to switch to another service",  "Product inquiry"),
    ("Is there a mobile application available for iOS and Android",       "Product inquiry"),
    ("What languages does the platform support for international users",  "Product inquiry"),
    ("Can I try the premium features before committing to a purchase",    "Product inquiry"),
    ("What are the terms of service and privacy policy for the product",  "Product inquiry"),
    ("How frequently are software updates released and are they free",    "Product inquiry"),
    ("I need to know the pricing details before making a decision",       "Product inquiry"),

    # Refund request
    ("I am extremely disappointed and I demand a full refund immediately","Refund request"),
    ("The product quality is terrible and completely not as described",   "Refund request"),
    ("This is unacceptable service and I want my money back right now",   "Refund request"),
    ("I received a damaged item and I need a refund or replacement",      "Refund request"),
    ("Your service did not work as promised and I want reimbursement",    "Refund request"),
    ("I am filing a complaint about the poor quality of this product",    "Refund request"),
    ("I was misled about the features and I want a full refund",          "Refund request"),
    ("The product stopped working after two days and I want money back",  "Refund request"),
    ("I am very unhappy with the service quality and want a refund",      "Refund request"),
    ("This product is defective and does not match the description",      "Refund request"),
    ("I never received my order and I want an immediate refund",          "Refund request"),
    ("The quality is far below what was advertised and I want refund",    "Refund request"),
    ("I am lodging a formal complaint due to unsatisfactory service",     "Refund request"),
    ("I waited three weeks and the product still has not arrived",        "Refund request"),
    ("I want to dispute this charge and receive a full reimbursement",    "Refund request"),
    ("The product malfunctioned on the first day of use",                 "Refund request"),
    ("I am deeply unsatisfied and request a complete refund of my money", "Refund request"),
    ("Your customer service is terrible and I want compensation",         "Refund request"),
    ("I received the wrong item and want a refund not a replacement",     "Refund request"),
    ("This purchase was a complete waste of money and I am complaining",  "Refund request"),
    ("The service was interrupted for days with no compensation offered", "Refund request"),
    ("I feel deceived by the false advertising and want my money back",   "Refund request"),
    ("The subscription did not include what was listed on the website",   "Refund request"),
    ("I am requesting reimbursement for a service that was never used",   "Refund request"),
    ("The product broke after one use and I want an immediate refund",    "Refund request"),

    # Cancellation request
    ("I want to cancel my subscription and stop all future payments",     "Cancellation request"),
    ("Please close my account and remove all my personal information",    "Cancellation request"),
    ("I wish to terminate my contract effective immediately",             "Cancellation request"),
    ("I want to unsubscribe from all services and delete my account",     "Cancellation request"),
    ("Please cancel my membership and confirm via email when done",       "Cancellation request"),
    ("I no longer need this service and want to end my subscription",     "Cancellation request"),
    ("Cancel my account right now and stop charging my credit card",      "Cancellation request"),
    ("I want to opt out of the service and discontinue my membership",    "Cancellation request"),
    ("Please process the cancellation of my account as soon as possible", "Cancellation request"),
    ("I have decided to stop using the service and want to cancel today", "Cancellation request"),
    ("End my subscription immediately and provide a cancellation number", "Cancellation request"),
    ("I want to deactivate my account and withdraw from the service",     "Cancellation request"),
    ("Please stop my subscription renewal and cancel all pending charges","Cancellation request"),
    ("I want my account permanently deleted and subscription cancelled",  "Cancellation request"),
    ("Cancel my plan and ensure I am not billed again in the future",     "Cancellation request"),
    ("I am moving to a competitor and need to cancel my current plan",    "Cancellation request"),
    ("Please remove me from the system and cancel the auto renewal",      "Cancellation request"),
    ("I want to close my account and receive a confirmation of closure",  "Cancellation request"),
    ("Terminate my service agreement effective from today please",        "Cancellation request"),
    ("I want to cancel before the next billing cycle to avoid charges",   "Cancellation request"),
    ("Stop the subscription immediately I do not wish to continue",       "Cancellation request"),
    ("Please deactivate all services linked to my account",               "Cancellation request"),
    ("I am requesting account termination and full data deletion",        "Cancellation request"),
    ("Cancel my trial before it converts to a paid subscription",        "Cancellation request"),
    ("I want to end my service contract and stop recurring payments",     "Cancellation request"),
]


# ── 3. Preprocesar todos los textos ───────────────────────────────────────────
def preprocess_all(X_texts, preprocessor):
    return [preprocessor.preprocess(text) for text in X_texts]


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("INICIO DEL ENTRENAMIENTO — Naive Bayes Multinomial")
    print("=" * 60)

    # Ruta al dataset (relativa a la raiz del proyecto)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH  = os.path.join(BASE_DIR, "data", DATASET_FILE)
    MODEL_PATH = os.path.join(BASE_DIR, "model", "naive_bayes_model.pkl")

    # 1. Cargar datos
    print(f"\n[1/5] Cargando dataset desde: {DATA_PATH}")
    X_texts, y = load_dataset(DATA_PATH)
    print(f"      Total de tickets cargados: {len(X_texts)}")

    from collections import Counter
    dist = Counter(y)
    print("      Distribucion por categoria:")
    for cat, cnt in dist.most_common():
        print(f"        {cat:<25} {cnt:>5} ({cnt/len(y)*100:.1f}%)")

    # 2. Mezclar aleatoriamente (importante para K-Folds)
    combined = list(zip(X_texts, y))
    random.seed(42)
    random.shuffle(combined)
    X_texts, y = zip(*combined)
    X_texts, y = list(X_texts), list(y)

    # 4. Preprocesar
    print("\n[2/5] Preprocesando texto (limpieza, tokenizacion, stopwords, stemming)...")
    preprocessor = Preprocessor()
    X_tokens = preprocess_all(X_texts, preprocessor)
    print(f"      Ejemplo: '{X_texts[0][:60]}...'")
    print(f"      Tokens : {X_tokens[0][:10]}...")

    classes = sorted(set(y))
    print(f"\n      Clases encontradas: {classes}")

    # 4. K-Folds Cross Validation (K=5)
    print("\n[3/5] Ejecutando K-Folds Cross Validation (K=5)...")
    kf = KFoldsEvaluator(k=5)
    folds = kf.split(X_tokens, y)

    fold_results = []
    all_y_true   = []   # Acumulamos predicciones de todos los folds
    all_y_pred   = []   # para la matriz de confusion global

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/5 — entrenando con {len(X_train)} docs, evaluando con {len(X_test)} docs...")

        # Construir vocabulario solo con datos de entrenamiento
        bow = BagOfWords()
        bow.fit(X_train)

        # Entrenar modelo
        clf = NaiveBayesClassifier(alpha=1.0)
        clf.fit(X_train, y_train)

        # Predecir sobre el fold de prueba
        y_pred = clf.predict(X_test)

        # Calcular metricas
        metrics, accuracy, macro_f1 = kf.compute_metrics(y_test, y_pred, classes)
        fold_results.append({
            'metrics':  metrics,
            'accuracy': accuracy,
            'macro_f1': macro_f1
        })

        # Acumular para matriz de confusion global
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # 5. Mostrar resultados completos
    print("\n[4/5] Calculando metricas finales...")
    kf.print_results(fold_results, classes, all_y_true, all_y_pred)

    # 6. Entrenar modelo FINAL con todos los datos
    print("\n[5/5] Entrenando modelo final con el dataset completo...")
    bow_final = BagOfWords()
    bow_final.fit(X_tokens)

    final_clf = NaiveBayesClassifier(alpha=1.0)
    final_clf.fit(X_tokens, y)

    # 7. Calcular métricas promedio por clase (sobre los K folds)
    avg_metrics = {}
    for clase in classes:
        avg_metrics[clase] = {
            'precision': round(sum(r['metrics'][clase]['precision'] for r in fold_results) / len(fold_results), 4),
            'recall':    round(sum(r['metrics'][clase]['recall']    for r in fold_results) / len(fold_results), 4),
            'f1':        round(sum(r['metrics'][clase]['f1']        for r in fold_results) / len(fold_results), 4),
        }

    avg_accuracy = round(sum(r['accuracy'] for r in fold_results) / len(fold_results), 4)
    avg_macro_f1 = round(sum(r['macro_f1'] for r in fold_results) / len(fold_results), 4)

    # Distribución de documentos por clase
    from collections import Counter as _Counter
    dist = _Counter(y)

    # 8. Guardar modelo + métricas
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_data = {
        'classifier':   final_clf,
        'preprocessor': preprocessor,
        'classes':      classes,
        # Métricas de evaluación (K-Folds)
        'eval': {
            'k':            5,
            'accuracy':     avg_accuracy,
            'macro_f1':     avg_macro_f1,
            'per_class':    avg_metrics,
            'n_docs':       len(X_tokens),
            'vocab_size':   final_clf.vocab_size,
            'docs_per_class': dict(dist),
        }
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n Modelo guardado en: {MODEL_PATH}")
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETO")
    print("Ahora ejecuta:  python web/app.py")
    print("=" * 60)
