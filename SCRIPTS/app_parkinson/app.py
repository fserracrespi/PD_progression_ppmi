import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DEFINICIÓN DE TU MODELO PYTORCH
# ==========================================
import torch.nn as nn

# IMPORTANTE: Debes pegar aquí la misma clase de tu modelo que usaste al entrenar
class TuModeloCascade(nn.Module):
    def __init__(self):
        super(TuModeloCascade, self).__init__()
        # ... tus capas ...
        pass
        
    def forward(self, x):
        # ... tu lógica forward ...
        return x

# ==========================================
# 2. CONFIGURACIÓN INICIAL Y CARGA
# ==========================================
# Instanciar y cargar el modelo
try:
    modelo = TuModeloCascade()
    modelo.load_state_dict(torch.load('modelo_cascade.pth'))
    modelo.eval() # Modo evaluación
    
    # Cargar el fondo para SHAP
    df_fondo = pd.read_csv('fondo_shap.csv')
    tensor_fondo = torch.tensor(df_fondo.values, dtype=torch.float32)
    
    # Inicializar el explicador de PyTorch
    explainer = shap.DeepExplainer(modelo, tensor_fondo)
    
except Exception as e:
    print(f"Error crítico en la inicialización: {e}")
    modelo = None

# ==========================================
# 3. LÓGICA DE LA APLICACIÓN
# ==========================================
def cargar_y_predecir():
    if modelo is None:
        messagebox.showerror("Error", "El modelo o los datos de fondo no se cargaron correctamente.")
        return

    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar CSV del Paciente",
        filetypes=(("Archivos CSV", "*.csv"), ("Todos", "*.*"))
    )
    
    if not ruta_archivo:
        return

    try:
        # Leer el paciente (1 fila)
        df_paciente = pd.read_csv(ruta_archivo)
        
        # Convertir a tensor de PyTorch
        tensor_paciente = torch.tensor(df_paciente.values, dtype=torch.float32)
        
        # Predicción
        with torch.no_grad(): # Para predecir no necesitamos gradientes
            prediccion_raw = modelo(tensor_paciente)
            # Adaptar según tu salida (ej: sigmoid, argmax, etc.)
            # prediccion = torch.sigmoid(prediccion_raw).item() 
            prediccion = prediccion_raw.numpy()[0]
            
        label_resultado.config(text=f"Resultado Predicción:\n{prediccion}")
        messagebox.showinfo("Éxito", "Predicción realizada. Generando explicación del paciente...")

        # Explicabilidad SHAP
        # DeepExplainer necesita el tensor con gradientes habilitados si el modelo lo requiere, 
        # pero para el cálculo de shap values lo maneja internamente.
        shap_values = explainer.shap_values(tensor_paciente)

        # PyTorch DeepExplainer a veces devuelve una lista si hay múltiples clases
        if isinstance(shap_values, list):
            # Tomamos la clase de interés (ej: clase 1)
            valores_explicar = shap_values[1][0] 
        else:
            valores_explicar = shap_values[0]

        # Gráfico de barras para 1 solo paciente
        plt.figure(figsize=(10, 6))
        
        # Usamos un gráfico de barras clásico para mostrar las variables de este paciente
        shap.bar_plot(valores_explicar, feature_names=df_paciente.columns.tolist(), show=False)
        
        plt.title("Impacto de las variables en este paciente específico")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error:\n{str(e)}")

# ==========================================
# 4. INTERFAZ GRÁFICA (TKINTER)
# ==========================================
root = tk.Tk()
root.title("Evaluación Clínica (Cascade Transfer Learning)")
root.geometry("450x250")
root.config(padx=20, pady=20)

tk.Label(root, text="Evaluador de Pacientes", font=("Arial", 16, "bold")).pack(pady=10)
tk.Label(root, text="Carga el archivo CSV con los datos del paciente.").pack(pady=5)

btn_cargar = tk.Button(root, text="Cargar CSV y Evaluar Paciente", command=cargar_y_predecir, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn_cargar.pack(pady=15)

label_resultado = tk.Label(root, text="", font=("Arial", 12), fg="darkblue")
label_resultado.pack(pady=5)

root.mainloop()