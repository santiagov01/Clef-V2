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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import optuna
import my_models

# reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


def configurar_logs(output_dir, timestamp):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'log_{timestamp}.txt')
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    return log


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
    # CSVs de train/val/test
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"), dtype={"label": str})
    val_df   = pd.read_csv(os.path.join(BASE_DIR, "val.csv"),   dtype={"label": str})
    test_df  = pd.read_csv(os.path.join(BASE_DIR, "test.csv"),  dtype={"label": str})
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Uno o más archivos CSV están vacíos.")

    def extraer(df):
        X = df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
        y = df["label"].values
        return X, y

    X_train, y_train = extraer(train_df)
    X_val,   y_val   = extraer(val_df)
    X_test,  y_test  = extraer(test_df)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)
    num_classes = len(le.classes_)

    # reshape para CNN/MLP
    X_train = X_train.reshape(-1, 1, 32, 96)
    X_val   = X_val.reshape(-1,   1, 32, 96)
    X_test  = X_test.reshape(-1,  1, 32, 96)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_enc))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val_enc))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test_enc))

    return train_ds, val_ds, test_ds, le, num_classes


def entrenar_y_evaluar(model, train_dl, val_dl, device, lr, epochs):
    """Entrena y retorna accuracy sobre val_dl."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluación final
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)
    return accuracy_score(y_true, y_pred)


def objective(trial, args, device, train_ds, val_ds, num_classes):
    # sugerir hiperparámetros
    lr        = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    n_layers  = trial.suggest_int("n_layers", 1, 4)
    n_neurons = trial.suggest_int("n_neurons", 32, 512)
    epochs    = trial.suggest_int("epochs", 5, 50)

    # DataLoaders con batch fijo
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=32)

    # construir MLP
    input_size   = 32 * 96
    hidden_sizes = [n_neurons] * n_layers
    model = my_models.get_MLP(num_classes=num_classes,
                              input_size=input_size,
                              hidden_sizes=hidden_sizes,
                              activation_fn=nn.ReLU)
    model.to(device)

    # entrenar y evaluar
    acc = entrenar_y_evaluar(model, train_dl, val_dl, device, lr, epochs)
    return acc


def entrenar_final_and_save_gpt(best_params, args, device, train_ds, test_ds, le):
    """Usa los mejores parámetros para entrenar en train+val y eval en test, guarda artefactos."""
    # unir train + val
    combined = torch.utils.data.ConcatDataset([train_ds])
    train_dl = DataLoader(combined, batch_size=32, shuffle=True)
    test_dl  = DataLoader(test_ds, batch_size=32)

    # crear modelo con mejores hiperparámetros
    hidden_sizes = [best_params["n_neurons"]] * best_params["n_layers"]
    input_size = 32 * 96
    model = my_models.get_MLP(num_classes=len(le.classes_),
                              input_size=input_size,
                              hidden_sizes=hidden_sizes,
                              activation_fn=nn.ReLU)
    model.to(device)

    # logs
    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    out_dir = os.path.join(args.output, f"run_{timestamp}")
    log = configurar_logs(out_dir, timestamp)
    log_hardware(log)

    # entrenamiento completo
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    for epoch in range(1, best_params["epochs"] + 1):
        model.train()
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(train_dl)
        train_losses.append(avg)
        log(f"Época {epoch}: Train Loss = {avg:.4f}")

    # evaluación en test
    log("\nEvaluando en TEST:")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)

    # reporte y matriz
    report = classification_report(y_true, y_pred,
                                   target_names=[str(c) for c in le.classes_],
                                   zero_division=0)
    log(report)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in le.classes_])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Matriz de Confusión Test")
    plt.savefig(os.path.join(out_dir, "confusion_test.png"))
    plt.close()
    log("Matriz de confusión guardada.")

    # guardar artefactos
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    log("Modelo y encoder guardados.")

def entrenar_final_and_save(best_params, args, device, train_ds, val_ds, test_ds, val_interval, le):
    """Usa los mejores parámetros para entrenar en train+val y eval en test, guarda artefactos."""
    # unir train + val
    #combined = torch.utils.data.ConcatDataset([train_ds])
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    test_dl  = DataLoader(test_ds, batch_size=32)

    # crear modelo con mejores hiperparámetros
    hidden_sizes = [best_params["n_neurons"]] * best_params["n_layers"]
    input_size = 32 * 96
    model = my_models.get_MLP(num_classes=len(le.classes_),
                              input_size=input_size,
                              hidden_sizes=hidden_sizes,
                              activation_fn=nn.ReLU)
    model.to(device)

    # logs
    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    out_dir = os.path.join(args.output, f"run_{timestamp}")
    log = configurar_logs(out_dir, timestamp)
    log_hardware(log)

    # entrenamiento completo
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    val_epochs = []

    for epoch in range(1, best_params["epochs"] + 1):
        # ====Entrenamiento=======
        model.train()
        total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_dl)
        train_losses.append(avg)
        log(f"Época {epoch}: Train Loss = {avg:.4f}")

        # ====Validación========
        if epoch % val_interval == 0 or epoch == best_params["epochs"]:
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
                torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
                log("Modelo mejorado guardado.")
    # ==== Graficar pérdidas ====
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, best_params["epochs"]  + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(val_epochs, val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title("Pérdida por época")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(out_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    log(f"Gráfico de pérdidas guardado en: {loss_plot_path}")

    # ======= evaluación en test ==========
    log("\nEvaluando en TEST:")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)

    # reporte y matriz
    report = classification_report(y_true, y_pred,
                                   target_names=[str(c) for c in le.classes_],
                                   zero_division=0)
    log(report)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(c) for c in le.classes_])
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Matriz de Confusión Test")
    plt.savefig(os.path.join(out_dir, "confusion_test.png"))
    plt.close()
    log("Matriz de confusión guardada.")

    # guardar artefactos
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    log("Modelo y encoder guardados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento con búsqueda Optuna de MLP/CNN")
    parser.add_argument("--csv",     type=str, default="embeddings_csv",
                        help="Carpeta con train.csv, val.csv, test.csv")
    parser.add_argument("--output",  type=str, default="outputs",
                        help="Directorio base de salida")
    parser.add_argument("--model",   type=str, default="mlp",
                        help="Tipo de modelo: 'mlp' (soporta Optuna) o cualquier CNN de my_models")
    parser.add_argument("--optuna",  action="store_true",
                        help="Si se activa, ejecuta búsqueda Optuna (solo para MLP)")
    parser.add_argument("--trials",  type=int, default=30,
                        help="Número de trials de Optuna")
    args = parser.parse_args()

    # carga datos
    train_ds, val_ds, test_ds, le, num_classes = cargar_datos_separados(args.csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.optuna and args.model.lower() == "mlp":
        # definir estudio y optimizar
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, args, device, train_ds, val_ds, num_classes),
                       n_trials=args.trials)

        print("===== Mejor trial =====")
        print(f"Accuracy: {study.best_value:.4f}")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

        # entrenar final y guardar
        entrenar_final_and_save(study.best_trial.params, args, device,
                                train_ds=train_ds,
                                val_ds=val_ds,
                                test_ds=test_ds,
                                val_interval=1,
                                le=le)
    else:
        # ejecución estándar (sin Optuna)
        timestamp = datetime.datetime.now().strftime("%d_%H%M")
        out_dir = os.path.join(args.output, f"run_{timestamp}")
        os.makedirs(out_dir, exist_ok=True)
        log = configurar_logs(out_dir, timestamp)
        log_hardware(log)

        # construir modelo
        if args.model.lower() == "mlp":
            hidden_sizes = [256, 128, 64]
            model = my_models.get_MLP(num_classes=num_classes,
                                      input_size=32*96,
                                      hidden_sizes=hidden_sizes,
                                      activation_fn=nn.ReLU)
        else:
            model = my_models.get_model(num_classes=num_classes, model_name=args.model)

        model.to(device)
        log(f"Modelo cargado y movido a {device}")

        # entrenar y guardar
        entrenar_y_evaluar(model,
                          DataLoader(train_ds, batch_size=32, shuffle=True),
                          DataLoader(val_ds,   batch_size=32),
                          device,
                          lr=1e-3,
                          epochs=20)
        evaluar_modelo = None  # si quieres reutilizar la función de evaluación original
        # … aquí podrías llamar a tu función evaluar_modelo() y guardar artefactos como antes.
# ejemplo de uso
# python train_TVT_optuna.py --csv embeddings_csv --output outputs --model mlp --optuna --trials 30
# python .\Train\train_TVT_optuna.py --csv .\embeddingscsv\ --output .\out_optuna_borrar\ --model mlp --optuna --trials 4