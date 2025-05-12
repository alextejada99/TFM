import os, time, csv, numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import onnxruntime as ort

# === CONFIG ===
test_dir   = "/home/atejada/Descargas/test"
onnx_model = "mobilenetv2_q8.onnx"
csv_out    = "resultados_onnx.csv"
classes    = [
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


preproc = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

sess     = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

results, y_true, y_pred = [], [], []
start = time.time()
for folder in tqdm(sorted(os.listdir(test_dir)), desc="Clases"):
    d = os.path.join(test_dir, folder)
    if not os.path.isdir(d): continue
    for fname in tqdm(sorted(os.listdir(d)), desc=folder, leave=False):
        if not fname.lower().endswith((".jpg",".jpeg",".png")): continue
        img = Image.open(os.path.join(d, fname)).convert("RGB")
        x   = preproc(img).unsqueeze(0).numpy()

        t0 = time.time()
        logits = sess.run(None, {inp_name: x})[0]
        latency = time.time() - t0

        idx  = int(np.argmax(logits, axis=1)[0])
        exps = np.exp(logits)
        probs= exps / np.sum(exps, axis=1, keepdims=True)
        pred = classes[idx]

        results.append([fname, folder, pred, f"{probs[0,idx]:.4f}", f"{latency:.3f}s"])
        y_true.append(folder); y_pred.append(pred)

# Guardar y métricas
with open(csv_out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["archivo","true","pred","conf","latency"])
    w.writerows(results)

print(classification_report(y_true, y_pred, labels=classes, zero_division=0, digits=4))
print(f"Accuracy: {accuracy_score(y_true,y_pred):.4f}")
print(f"Tiempo total: {time.time()-start:.1f}s")
