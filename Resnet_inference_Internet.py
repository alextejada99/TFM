import os
import random
import time
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report

# === CONFIG ===
test_dir    = "/home/atejada/Imágenes/testing"
modelo_pt   = "resnet50_traced.pt"   # o "mobilenetv2_pi_traced.pt"
output_csv  = "resultados_validacion_noOOD_resnet.csv"

# === CLASES ENTRENADAS ===
class_names = [
    "Apple_Black_rot","Apple_Cedar_apple_rust","Apple_healthy","Apple_scab",
    "Blueberry_healthy","Cherry_Powdery_mildew","Cherry_healthy",
    "Corn_Cercospora_leaf_spot_Gray_leaf_spot","Corn_Common_rust",
    "Corn_Northern_Leaf_Blight","Corn_healthy","Grape_Black_rot",
    "Grape_Esca_Black_Measles","Grape_Leaf_blight_Isariopsis_Leaf_Spot",
    "Grape_healthy","Orange_Haunglongbing_Citrus_greening",
    "Peach_Bacterial_spot","Peach_healthy","Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy","Potato_Early_blight","Potato_Late_blight",
    "Potato_healthy","Raspberry_healthy","Soybean_healthy",
    "Squash_Powdery_mildew","Strawberry_Leaf_scorch","Strawberry_healthy",
    "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight",
    "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites",
    "Tomato_Target_Spot","Tomato_Yellow_Leaf_Curl_Virus","Tomato_healthy",
    "Tomato_mosaic_virus"
]

# === PREPROCESADO ===
preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# === CARGA MODELO TorchScript ===
model = torch.jit.load(modelo_pt, map_location="cpu")
model.eval()

# === INFERENCIA ===
rows = []; y_true=[]; y_pred=[]
start = time.time()

for fname in sorted(os.listdir(test_dir)):
    if not fname.lower().endswith((".jpg",".jpeg",".png")):
        continue

    true_label = fname.rsplit("_",1)[0]
    if true_label not in class_names:
        continue  # saltar OOD

    img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
    x   = preproc(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx  = int(np.argmax(probs))
    conf = float(probs[idx])
    pred = class_names[idx]

    rows.append([fname, true_label, pred, f"{conf:.4f}"])
    y_true.append(true_label)
    y_pred.append(pred)

# Guardar CSV
with open(output_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["archivo","true","pred","confidence"])
    w.writerows(rows)

# Métricas
labels_present = sorted(set(y_true))
print(f"\nAccuracy global: {accuracy_score(y_true, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(
    y_true, y_pred,
    labels=labels_present,
    zero_division=0,
    digits=4
))

# Muestra aleatoria
sample = random.sample(rows, 4)
fig, axes = plt.subplots(1,4,figsize=(16,4))
for ax, (fname, true_label, pred, conf) in zip(axes, sample):
    img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
    ax.imshow(img); ax.axis("off")
    color = "green" if true_label==pred else "red"
    ax.set_title(f"T:{true_label}\nP:{pred}\nConf:{conf}", color=color)
plt.tight_layout()
plt.show()

print(f"\nTiempo total: {time.time()-start:.1f}s")
