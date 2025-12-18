import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import os
import time
import multiprocessing
from datetime import datetime

# Импорт ваших классов
from data_loader import Br35HDataset
from gnn_model import BrainTumorGNN

# --- ПАРАМЕТРЫ ---
BEST_PARAMS = {
    'learning_rate': 0.00215,
    'hidden_channels': 128,
    'dropout': 0.128,
    'weight_decay': 9.67e-05,
    'optimizer': 'AdamW'
}

EPOCHS = 150
BATCH_SIZE = 32
PATIENCE = 20
DATA_ROOT = '.'
LOG_FILE = "training_log.txt"
# НАСТРОЙКА ПОТОКОВ (Важно для Windows)
# Попробуйте 4. Если вылетает ошибка - поставьте 0.
NUM_WORKERS = 0 

def log_msg(message):
    """Функция пишет сообщение и в консоль, и в файл"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    
    print(full_msg) # Вывод на экран
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n") # Запись в файл

def train_full_model():
    # Очистка лога при новом запуске
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== STARTING TRAINING SESSION ===\n")

    log_msg(f"🚀 ЗАПУСК ОБУЧЕНИЯ (Logging Enabled)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"💻 Устройство: {device}")
    
    if torch.cuda.is_available():
        log_msg(f"ℹ️ GPU: {torch.cuda.get_device_name(0)}")

    log_msg(f"⚙️ Workers: {NUM_WORKERS} | Batch Size: {BATCH_SIZE}")

    log_msg("📂 Загрузка датасета...")
    try:
        train_dataset = Br35HDataset(root=os.path.join(DATA_ROOT, 'br35h'), split='train', n_segments=100)
        test_dataset = Br35HDataset(root=os.path.join(DATA_ROOT, 'br35h'), split='test', n_segments=100)
    except Exception as e:
        log_msg(f"❌ Ошибка загрузки данных: {e}")
        return

    # DataLoader с оптимизацией
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,      
        pin_memory=True,          
        persistent_workers=(NUM_WORKERS > 0) 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    log_msg("🧠 Инициализация модели...")
    model = BrainTumorGNN(
        input_dim=train_dataset[0].x.shape[1],
        hidden_dim=BEST_PARAMS['hidden_channels'],
        num_classes=2,
        dropout=BEST_PARAMS['dropout']
    ).to(device)

    # Оптимизатор
    if BEST_PARAMS['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['learning_rate'], weight_decay=BEST_PARAMS['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=BEST_PARAMS['learning_rate'], weight_decay=BEST_PARAMS['weight_decay'])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    
    log_msg(f"▶️ Старт цикла обучения на {EPOCHS} эпох...")
    
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        start_epoch = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Tqdm только для визуала в консоли (leave=False, чтобы не засорять)
        loop = tqdm(train_loader, leave=False, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y.squeeze()).sum())
            total += batch.y.size(0)
            
            loop.set_postfix(loss=loss.item())
        
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_accum = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.squeeze())
                val_loss_accum += loss.item()
                pred = out.argmax(dim=1)
                val_correct += int((pred == batch.y.squeeze()).sum())
                val_total += batch.y.size(0)
                
                y_true.extend(batch.y.squeeze().cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        val_acc = val_correct / val_total
        val_loss = val_loss_accum / len(test_loader)
        
        epoch_time = time.time() - start_epoch
        
        # ЗАПИСЬ В ЛОГ
        log_msg(f"Epoch {epoch+1:03d}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Обновление LR
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            log_msg(f"ℹ️ LR changed from {old_lr} to {new_lr}")

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'br35h_best_baseline.pth')
            best_y_true = y_true
            best_y_pred = y_pred
            # log_msg(f"★ New best model saved! Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log_msg(f"⏹️ Early stopping triggered after {PATIENCE} epochs without improvement.")
                break

    total_time = (time.time() - total_start_time) / 60
    log_msg(f"🏁 Обучение завершено за {total_time:.2f} минут.")
    log_msg(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Сохранение графиков
    log_msg("📊 Генерация графиков...")
    save_plots(history, best_y_true, best_y_pred, ['No Tumor', 'Tumor'])
    log_msg("✅ Все готово!")

def save_plots(history, y_true, y_pred, class_names):
    sns.set_style("whitegrid")
    
    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig('br35h_accuracy_curve.png')
    
    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('br35h_loss_curve.png')

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig('br35h_confusion_matrix.png')
    
    # Report to file
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open("br35h_final_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_full_model()