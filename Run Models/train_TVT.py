import os
import argparse
import datetime
import time
import platform
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import my_models

torch.manual_seed(42)
np.random.seed(42)

def configurar_logs(output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'log_{timestamp}.txt')
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    return log, log_file

def log_hardware(log):
    log("\nResumen del hardware:")
    log(f"  Plataforma: {platform.system()} {platform.release()}")
    log(f"  Procesador: {platform.processor()}")
    log(f"  PyTorch version: {torch.__version__}")
    log(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log("")


def cargar_datos_separados(BASE_DIR):
    
    # Dirs
    train_csv = os.path.join(BASE_DIR, "train.csv")
    val_csv = os.path.join(BASE_DIR, "val.csv")
    test_csv = os.path.join(BASE_DIR, "test.csv")

    # Load train data
    train_df = pd.read_csv(train_csv, dtype={"label": str})
    X_train = train_df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
    y_train = train_df["label"].values

    # Load validation data
    val_df = pd.read_csv(val_csv, dtype={"label": str})
    X_val = val_df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
    y_val = val_df["label"].values

    # Load test data
    test_df = pd.read_csv(test_csv, dtype={"label": str})
    X_test = test_df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
    y_test = test_df["label"].values

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    num_classes = len(le.classes_)

    # Reshape data
    X_train = X_train.reshape(-1, 1, 32, 96)
    X_val = X_val.reshape(-1, 1, 32, 96)
    X_test = X_test.reshape(-1, 1, 32, 96)

    # Create TensorDatasets
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_encoded))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val_encoded))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test_encoded))

    return train_ds, val_ds, test_ds, le, num_classes

def entrenar_modelo(model, train_dl, val_dl, device, log, epochs=20, val_interval=1, output_dir="."):
    """
    Entrena el modelo y evalúa en el conjunto de validación cada `val_interval` épocas. 
    Guarda un gráfico de pérdidas al final.

    Args:
        model: El modelo a entrenar.
        train_dl: DataLoader para entrenamiento.
        val_dl: DataLoader para validación.
        device: CPU o GPU.
        log: Función para imprimir y guardar log.
        epochs: Número de épocas.
        val_interval: Cada cuántas épocas se evalúa el modelo.
        output_dir: Carpeta donde guardar el gráfico.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    val_epochs = []

    for epoch in range(1, epochs + 1):
        # Entrenamiento
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        train_losses.append(avg_loss)
        log(f"Época {epoch}: Train Loss = {avg_loss:.4f}")

        # Validación
        if epoch % val_interval == 0 or epoch == epochs:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device).long()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dl)
            val_losses.append(avg_val_loss)
            val_epochs.append(epoch)
            log(f"Época {epoch}: Validation Loss = {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                log("Modelo mejorado guardado.")

    # ==== Graficar pérdidas ====
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(val_epochs, val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title("Pérdida por época")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    log(f"Gráfico de pérdidas guardado en: {loss_plot_path}")

def evaluar_modelo(model, val_dl, le, device, output_dir, log, timestamp, title=""):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)
    labels = list(range(len(le.classes_)))
    target_names = [str(c) for c in le.classes_]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    log("\nReporte de resultados:")
    log(report)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title(f"Matriz de Confusion {title}")
    plt.grid(False)
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}_{title}.png")
    plt.savefig(cm_path)
    plt.close()
    log(f"\nMatriz de confusion guardada como '{cm_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador CNN con embeddings BirdNET")
    parser.add_argument("--csv", default="embeddings_csv", help="Archivo CSV de entrenamiento")
    parser.add_argument("--output", type=str, default="outputs", help="Directorio base de salida")
    parser.add_argument("--epochs", type=int, default=20, help="Número de epocas de entrenamiento")
    parser.add_argument("--model_type", type=str, default="resnet18", help="Tipo de modelo a usar: 'resnet18' o 'mlp'")
    parser.add_argument("--pretrained", type=int, default=True, help="Usar pesos preentrenados (1) o no (0)")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    run_output_dir = os.path.join(args.output, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    log, log_file = configurar_logs(run_output_dir, timestamp)
    log_hardware(log)

    train_ds, val_ds, test_ds, le, num_classes = cargar_datos_separados(args.csv)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)

    if args.model_type == "mlp":
        input_size = 32 * 96
        model = my_models.get_MLP(num_classes=num_classes, input_size=input_size, hidden_sizes=[256, 128, 64], activation_fn=nn.ReLU)
    else:
        model = my_models.get_model(num_classes=num_classes, model_name=args.model_type, pretrained=args.pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log(f"Modelo cargado y movido a {device}")
    start_time = time.time()
    #entrenar_modelo(model, train_dl, device, log, epochs=args.epochs)
    entrenar_modelo(model, train_dl, val_dl, device, log, epochs=args.epochs, output_dir=run_output_dir)
    elapsed_time = time.time() - start_time

    # Evaluate on test
    evaluar_modelo(model, test_dl, le, device, run_output_dir, log, timestamp, title="Test")

    model_path = os.path.join(run_output_dir, f"modelo_{args.model_type}.pt")
    encoder_path = os.path.join(run_output_dir, "label_encoder.pkl")
    torch.save(model.state_dict(), model_path)
    joblib.dump(le, encoder_path)
    log(f"\nModelo guardado en '{model_path}'")
    log(f"\nLabel encoder guardado en '{encoder_path}'")
    log(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos")

# ======================= EXECUTION========================
# DEFAULT EXECUTION 
# python Train_Inference/train_TVT.py --epochs 20 --model_type mlp --pretrained False

# default Args
# --csv embeddings_csv
# --output outputs
# --epochs 20
# --model_type resnet18
# --pretrained True