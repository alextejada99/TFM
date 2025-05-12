import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score
)

# === CONFIGURACIÓN ===
CSV_PATH = "resultados_onnx.csv"

# === CARGAR DATOS ===
df = pd.read_csv(CSV_PATH)

# (Si tu CSV usa otros nombres, descomenta y ajusta)
# df = df.rename(columns={"clase_real":"true", "clase_predicha":"pred"})

# === CLASES EVALUADAS ===
labels = sorted(df['true'].unique())

# === CÁLCULO DE MÉTRICAS ===
accuracy        = accuracy_score(df['true'], df['pred'])
precision_micro = precision_score(df['true'], df['pred'], average='micro', zero_division=0)
precision_macro = precision_score(df['true'], df['pred'], average='macro', zero_division=0)
f1_weighted     = f1_score(df['true'], df['pred'], average='weighted', zero_division=0)

print("===== Resumen de métricas =====")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precisión global   : {precision_micro:.4f}")
print(f"Precisión macro    : {precision_macro:.4f}")
print(f"F1-score ponderado : {f1_weighted:.4f}\n")

# === INFORME DE CLASIFICACIÓN ===
print("===== Classification Report =====")
print(classification_report(
    df['true'],
    df['pred'],
    labels=labels,
    zero_division=0,
    digits=4
))

# === MATRIZ DE CONFUSIÓN ===
cm = confusion_matrix(df['true'], df['pred'], labels=labels)
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax, ax=ax)
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center',
            color='white' if val > cm.max()/2 else 'black')
plt.tight_layout()
plt.show()

# === LATENCIA PROMEDIO POR CLASE (si la guardaste) ===
if 'latency' in df.columns:
    df['latency_s'] = df['latency'].str.rstrip('s').astype(float)
    avg_lat = df.groupby('true')['latency_s'].mean().loc[labels]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(avg_lat.index, avg_lat.values)
    ax2.set_xlabel('Average Latency (s)')
    ax2.set_title('Average Inference Time per Class')
    plt.tight_layout()
    plt.show()
