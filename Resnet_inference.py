import os
import torch
import pickle
import time
import csv
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# === CONFIGURACIÓN ===
directorio_imagenes = "/home/atejada/Descargas/test"
modelo_path = "/home/atejada/Descargas/resnet50_traced.pt"
#UMBRAL_CONFIANZA = 0.70
archivo_csv = "resultados_tomate.csv"
archivo_pickle = "imagenes_info_tomate.pkl"

# === CLASES COMPLETAS DEL MODELO (38) ===
class_names = [
    "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy", "Apple_scab",
    "Blueberry_healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
    "Corn_Cercospora_leaf_spot", "Corn_Common_rust", "Corn_Northern_Leaf_Blight", "Corn_healthy",
    "Grape_Black_rot", "Grape_Esca_Black_Measles", "Grape_Leaf_blight", "Grape_healthy",
    "Orange_Haunglongbing_Citrus_greening", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
    "Raspberry_healthy", "Soybean_healthy", "Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_healthy", "Tomato_mosaic_virus"
]

# === RECOLECTAR IMÁGENES ===
imagenes_info = []
for carpeta in os.listdir(directorio_imagenes):
    clase = carpeta.replace(" ", "_")
    ruta_carpeta = os.path.join(directorio_imagenes, carpeta)
    if os.path.isdir(ruta_carpeta):
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                ruta_img = os.path.join(ruta_carpeta, archivo)
                imagenes_info.append({
                    "ruta": ruta_img,
                    "archivo": archivo,
                    "clase": clase
                })

# Guardar .pkl
with open(archivo_pickle, "wb") as f:
    pickle.dump(imagenes_info, f)
print(f"✅ {len(imagenes_info)} imágenes encontradas y guardadas en {archivo_pickle}")

# === TRANSFORMACIÓN ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CARGAR MODELO ===
model = torch.jit.load(modelo_path, map_location='cpu')
model.eval()

# === EVALUACIÓN ===
resultados = []
y_true, y_pred = [], []
start = time.time()

for item in tqdm(imagenes_info, desc="🔍 Evaluando imágenes"):
    ruta_img = item["ruta"]
    clase_real = item["clase"]
    archivo = item["archivo"]

    if clase_real not in class_names:
        continue

    image = Image.open(ruta_img).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()

        if pred_idx >= len(class_names):
            continue

        clase_predicha = class_names[pred_idx]

    estado = "Correcta" if clase_real == clase_predicha else "Incorrecta"
    resultados.append([archivo, clase_real, clase_predicha, f"{confidence:.4f}", estado])
    y_true.append(clase_real)
    y_pred.append(clase_predicha)

# === GUARDAR CSV ===
with open(archivo_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["archivo", "clase_real", "clase_predicha", "confianza", "estado"])
    writer.writerows(resultados)

# === MOSTRAR MÉTRICAS ===
print("\n===== MÉTRICAS DE CLASIFICACIÓN =====")
print(classification_report(y_true, y_pred, digits=4))
print(f"✅ Accuracy global: {accuracy_score(y_true, y_pred):.4f}")
print(f"📄 Resultados guardados en: {archivo_csv}")
print(f"⏱ Tiempo total: {time.time() - start:.2f} segundos")
