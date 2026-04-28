import math
from collections import defaultdict


class KFoldsEvaluator:
    """
    Implementa K-Folds Cross Validation y calcula métricas manualmente.
    """

    def __init__(self, k=5):
        self.k = k

    def split(self, X, y):
        """
        Divide los datos en K folds.
        Retorna lista de (X_train, y_train, X_test, y_test) para cada fold.
        """
        n = len(X)
        fold_size = n // self.k
        folds = []

        for i in range(self.k):
            start = i * fold_size
            end = start + fold_size if i < self.k - 1 else n

            X_test = X[start:end]
            y_test = y[start:end]
            X_train = X[:start] + X[end:]
            y_train = y[:start] + y[end:]

            folds.append((X_train, y_train, X_test, y_test))

        return folds

    def confusion_matrix(self, y_true, y_pred, classes):
        """
        Construye la matriz de confusion.
        Retorna dict {(real, predicho): count}
        """
        matrix = defaultdict(int)
        for real, pred in zip(y_true, y_pred):
            matrix[(real, pred)] += 1
        return matrix

    def compute_metrics(self, y_true, y_pred, classes):
        """
        Calcula Precision, Recall, F1-Score por clase y Accuracy global.
        """
        metrics = {}

        for clase in classes:
            tp = sum(1 for r, p in zip(y_true, y_pred) if r == clase and p == clase)
            fp = sum(1 for r, p in zip(y_true, y_pred) if r != clase and p == clase)
            fn = sum(1 for r, p in zip(y_true, y_pred) if r == clase and p != clase)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            metrics[clase] = {
                'precision': round(precision, 4),
                'recall':    round(recall,    4),
                'f1':        round(f1,        4),
                'tp': tp, 'fp': fp, 'fn': fn
            }

        correct  = sum(1 for r, p in zip(y_true, y_pred) if r == p)
        accuracy = correct / len(y_true)
        macro_f1 = sum(metrics[c]['f1'] for c in classes) / len(classes)

        return metrics, round(accuracy, 4), round(macro_f1, 4)

    def print_confusion_matrix(self, y_true, y_pred, classes):
        """Imprime la matriz de confusion en formato de tabla legible."""
        matrix = self.confusion_matrix(y_true, y_pred, classes)

        # Etiquetas cortas para que la tabla quepa en pantalla
        short = {
            "Technical issue":      "Tech",
            "Billing inquiry":      "Bill",
            "Product inquiry":      "Prod",
            "Refund request":       "Refu",
            "Cancellation request": "Canc",
        }
        labels = [short.get(c, c[:5]) for c in classes]
        col_w = 7

        print(f"\n{'='*60}")
        print("MATRIZ DE CONFUSION (filas=real, columnas=predicho)")
        print(f"{'='*60}")
        header = f"{'Real / Pred':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
        print(header)
        print("-" * len(header))
        for real, label in zip(classes, labels):
            row = f"{label:<20}"
            for pred in classes:
                row += f"{matrix.get((real, pred), 0):>{col_w}}"
            print(row)
        print(f"{'='*60}")
        print("-> Diagonal principal = predicciones correctas.")
        print("-> Fuera de diagonal  = errores de clasificacion.")
        print(f"{'='*60}")

    def print_results(self, fold_results, classes, all_y_true=None, all_y_pred=None):
        """Imprime resumen de todos los folds y la matriz de confusion global."""
        print("\n" + "="*60)
        print("RESULTADOS K-FOLDS CROSS VALIDATION")
        print("="*60)

        all_accuracies = [r['accuracy'] for r in fold_results]
        all_macro_f1   = [r['macro_f1'] for r in fold_results]

        for i, result in enumerate(fold_results):
            print(f"\nFold {i+1}: Accuracy={result['accuracy']:.4f} | Macro F1={result['macro_f1']:.4f}")

        avg_acc = sum(all_accuracies) / len(all_accuracies)
        avg_f1  = sum(all_macro_f1)   / len(all_macro_f1)
        var_acc = sum((x - avg_acc)**2 for x in all_accuracies) / len(all_accuracies)

        print(f"\n{'─'*40}")
        print(f"Accuracy promedio : {avg_acc:.4f} +/- {math.sqrt(var_acc):.4f}")
        print(f"Macro F1 promedio : {avg_f1:.4f}")

        # Metricas por clase promediadas entre los K folds
        print(f"\n{'─'*40}")
        print(f"METRICAS POR CLASE (promedio de {len(fold_results)} folds)")
        print(f"{'Clase':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"{'─'*55}")
        for clase in classes:
            avg_p = sum(r['metrics'][clase]['precision'] for r in fold_results) / len(fold_results)
            avg_r = sum(r['metrics'][clase]['recall']    for r in fold_results) / len(fold_results)
            avg_f = sum(r['metrics'][clase]['f1']        for r in fold_results) / len(fold_results)
            print(f"{clase:<25} {avg_p:>10.4f} {avg_r:>10.4f} {avg_f:>10.4f}")
        print("="*60)

        if all_y_true and all_y_pred:
            self.print_confusion_matrix(all_y_true, all_y_pred, classes)
