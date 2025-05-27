# TFM – Sistema para la detección de enfermedades en hojas de cultivo mediante visión por computadora y Deep Learning: optimización y evaluación en Raspberry Pi

Este repositorio contiene el código completo del Trabajo de Fin de Máster (TFM), centrado en la detección automática de enfermedades en hojas de cultivo mediante modelos de clasificación basados en visión por computadora. El sistema fue optimizado para ejecutarse en una **Raspberry Pi 5**, evaluando tanto imágenes reales de campo (tomate) como imágenes externas obtenidas desde Internet.

---

## 📂 Estructura del Repositorio

| Archivo | Descripción |
|--------|-------------|
| **MobileNet_inference.py** | Script optimizado para inferencia local en Raspberry Pi 5 usando MobileNetV2. Toma como entrada una carpeta con imágenes (dataset de tomate), genera predicciones y guarda los resultados en un archivo CSV. |
| **Mobilenetv2_inference_Internet.py** | Variante del script anterior, adaptado para inferencia sobre imágenes descargadas desde Internet. |
| **Mobilenetv2_summary.py** | Evalúa el rendimiento del modelo MobileNetV2. Toma como entrada un archivo CSV generado por los scripts de inferencia y calcula métricas como accuracy, precision, recall y F1-score. |
| **Mobilenetv2_training.ipynb** | Notebook de entrenamiento del modelo MobileNetV2. Incluye carga de datos, preprocesamiento, arquitectura, entrenamiento y validación. |
| **Resnet_inference.py** | Script optimizado para ejecutar inferencia local con ResNet-50 en Raspberry Pi 5 sobre el dataset de tomate. |
| **Resnet_inference_Internet.py** | Realiza inferencia con ResNet-50 sobre imágenes provenientes de Internet. |
| **ResNet_summary.py** | Evalúa el rendimiento del modelo ResNet-50 utilizando los resultados CSV obtenidos de los scripts de inferencia. |
| **ResNet-50_training.ipynb** | Notebook de entrenamiento del modelo ResNet-50. Incluye configuración, entrenamiento, evaluación y guardado del modelo. |
| **mobilenetv2_optimized.onnx** | Modelo MobileNetV2 optimizado en formato ONNX para inferencia eficiente. Compatible con ONNX Runtime en dispositivos edge. |
| **resnet50_optimized.pt** | Modelo ResNet-50 entrenado y exportado en formato `.pt`, listo para usar con PyTorch. |
| **Internet_dataset.7z** | Archivo comprimido que contiene imágenes recopiladas desde Internet para pruebas externas de inferencia. |

---

## 🚀 Flujo General del Sistema

1. **Entrenamiento del modelo**  
   Se realiza en Jupyter Notebooks (`Mobilenetv2_training.ipynb`, `ResNet-50_training.ipynb`).

2. **Inferencia**  
   - **Imágenes reales (tomate):**
     - `MobileNet_inference.py`
     - `Resnet_inference.py`
   - **Imágenes externas (Internet):**
     - `Mobilenetv2_inference_Internet.py`
     - `Resnet_inference_Internet.py`

3. **Evaluación de resultados**  
   - `Mobilenetv2_summary.py`
   - `ResNet_summary.py`  
   (Ambos procesan los CSV generados por los scripts de inferencia para calcular métricas de clasificación)


---
## 🛠️ Requisitos
Python ≥ 3.8

PyTorch, torchvision

scikit-learn

ONNX Runtime (si se usa .onnx)

PIL, NumPy, Pandas

---
## 🧪 Ejemplo de Inferencia

```bash
python MobileNet_inference.py /ruta/a/imagenes



