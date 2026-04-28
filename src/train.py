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

from config import DATASET_FILE, COL_TEXT, COL_LABEL


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



def preprocess_all(X_texts, preprocessor):
    return [preprocessor.preprocess(text) for text in X_texts]


if __name__ == "__main__":
    print("=" * 60)
    print("INICIO DEL ENTRENAMIENTO — Naive Bayes Multinomial")
    print("=" * 60)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH  = os.path.join(BASE_DIR, "data", DATASET_FILE)
    MODEL_PATH = os.path.join(BASE_DIR, "model", "naive_bayes_model.pkl")

    print(f"\n[1/5] Cargando dataset desde: {DATA_PATH}")
    X_texts, y = load_dataset(DATA_PATH)
    print(f"      Total de tickets cargados: {len(X_texts)}")

    from collections import Counter
    dist = Counter(y)
    print("      Distribucion por categoria:")
    for cat, cnt in dist.most_common():
        print(f"        {cat:<25} {cnt:>5} ({cnt/len(y)*100:.1f}%)")

    combined = list(zip(X_texts, y))
    random.seed(42)
    random.shuffle(combined)
    X_texts, y = zip(*combined)
    X_texts, y = list(X_texts), list(y)

    print("\n[2/5] Preprocesando texto (limpieza, tokenizacion, stopwords, stemming)...")
    preprocessor = Preprocessor()
    X_tokens = preprocess_all(X_texts, preprocessor)
    print(f"      Ejemplo: '{X_texts[0][:60]}...'")
    print(f"      Tokens : {X_tokens[0][:10]}...")

    classes = sorted(set(y))
    print(f"\n      Clases encontradas: {classes}")

    print("\n[3/5] Ejecutando K-Folds Cross Validation (K=5)...")
    kf = KFoldsEvaluator(k=5)
    folds = kf.split(X_tokens, y)

    fold_results = []
    all_y_true   = []   
    all_y_pred   = []   

    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/5 — entrenando con {len(X_train)} docs, evaluando con {len(X_test)} docs...")
        bow = BagOfWords()
        bow.fit(X_train)

        clf = NaiveBayesClassifier(alpha=1.0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        metrics, accuracy, macro_f1 = kf.compute_metrics(y_test, y_pred, classes)
        fold_results.append({
            'metrics':  metrics,
            'accuracy': accuracy,
            'macro_f1': macro_f1
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    print("\n[4/5] Calculando metricas finales...")
    kf.print_results(fold_results, classes, all_y_true, all_y_pred)

    print("\n[5/5] Entrenando modelo final con el dataset completo...")
    bow_final = BagOfWords()
    bow_final.fit(X_tokens)

    final_clf = NaiveBayesClassifier(alpha=1.0)
    final_clf.fit(X_tokens, y)

    avg_metrics = {}
    for clase in classes:
        avg_metrics[clase] = {
            'precision': round(sum(r['metrics'][clase]['precision'] for r in fold_results) / len(fold_results), 4),
            'recall':    round(sum(r['metrics'][clase]['recall']    for r in fold_results) / len(fold_results), 4),
            'f1':        round(sum(r['metrics'][clase]['f1']        for r in fold_results) / len(fold_results), 4),
        }

    avg_accuracy = round(sum(r['accuracy'] for r in fold_results) / len(fold_results), 4)
    avg_macro_f1 = round(sum(r['macro_f1'] for r in fold_results) / len(fold_results), 4)

    from collections import Counter as _Counter
    dist = _Counter(y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model_data = {
        'classifier':   final_clf,
        'preprocessor': preprocessor,
        'classes':      classes,
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
