"""
filter_dataset.py
-----------------
Lee el CSV crudo del Consumer Complaint Database (CFPB),
filtra solo las filas que tienen narrativa escrita por el consumidor,
selecciona las 5 categorias con mas datos y guarda el resultado
en data/complaints_filtered.csv listo para el entrenamiento.

Uso:
    python src/filter_dataset.py
"""

import csv
import os
from collections import Counter

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "complaints_raw.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "complaints_filtered.csv")


TARGET_CATEGORIES = [
    "Debt collection",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Mortgage",
    "Credit card or prepaid card",
    "Checking or savings account",
]

CATEGORY_MAP = {
    "Debt collection":          "Debt collection",
    "Credit reporting, credit repair services, or other personal consumer reports":
                                "Credit reporting",
    "Mortgage":                 "Mortgage",
    "Credit card or prepaid card": "Credit card",
    "Checking or savings account": "Bank account",
}

print("=" * 60)
print("FILTRADO DEL DATASET — Consumer Complaint Database")
print("=" * 60)
print(f"\nLeyendo: {INPUT_PATH}")

if not os.path.exists(INPUT_PATH):
    print("\n ERROR: No se encontro el archivo.")
    print(f"  Descarga el CSV de Kaggle y guardalo como:")
    print(f"  {INPUT_PATH}")
    exit(1)

total_leidos    = 0
sin_narrativa   = 0
categoria_wrong = 0
rows_filtradas  = []

with open(INPUT_PATH, encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)

    fieldnames = reader.fieldnames or []
    if 'Consumer complaint narrative' not in fieldnames:
        print(f"\n ERROR: No se encontro la columna 'Consumer complaint narrative'")
        print(f"  Columnas disponibles: {fieldnames}")
        exit(1)

    for row in reader:
        total_leidos += 1

        narrative = row.get('Consumer complaint narrative', '').strip()
        product   = row.get('Product', '').strip()

        if not narrative or narrative.lower() == 'nan':
            sin_narrativa += 1
            continue

        if product not in TARGET_CATEGORIES:
            categoria_wrong += 1
            continue

        rows_filtradas.append({
            'text':  narrative,
            'label': CATEGORY_MAP.get(product, product)
        })

print(f"  Total leidos       : {total_leidos:,}")
print(f"  Sin narrativa      : {sin_narrativa:,}")
print(f"  Otra categoria     : {categoria_wrong:,}")
print(f"  Filas utiles       : {len(rows_filtradas):,}")

dist = Counter(r['label'] for r in rows_filtradas)
print(f"\nDistribucion por categoria:")
for cat, cnt in dist.most_common():
    print(f"  {cat:<45} {cnt:>7,} ({cnt/len(rows_filtradas)*100:.1f}%)")

min_per_class = min(dist.values())
cap_per_class = min(min_per_class, 20_000)

print(f"\nBalanceando a {cap_per_class:,} ejemplos por categoria...")

from collections import defaultdict
import random

by_cat = defaultdict(list)
for row in rows_filtradas:
    by_cat[row['label']].append(row)

random.seed(42)
balanced = []
for cat, items in by_cat.items():
    random.shuffle(items)
    balanced.extend(items[:cap_per_class])

random.shuffle(balanced)

print(f"Total final        : {len(balanced):,} filas")

with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['text', 'label'])
    writer.writeheader()
    writer.writerows(balanced)

print(f"\n Guardado en: {OUTPUT_PATH}")
print("\nAhora ejecuta:  python src/train.py")
print("=" * 60)
