import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from data_loader import ECGDataLoader
from sklearn.metrics import accuracy_score

NUM_CLASSES = 5
NUM_PATIENTS = 48
MODEL_SAVE_PATH = 'saved_models'
RESULT_PATH = 'results'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
SEG_LENGTH = 5
RANDOM_STATE = 999
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
FEATURE_DIM = 256
D_K = 64

FINAL = str(RANDOM_STATE) + '_ican_final.pth'
BEST = str(RANDOM_STATE) + '_ican_best.pth'

def prepare_dataloaders(data):
    X_train, y_train, ids_train = data['train']
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(1),
        torch.LongTensor(y_train),
        torch.LongTensor(ids_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_val, y_val, ids_val = data['val']
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).unsqueeze(1),
        torch.LongTensor(y_val),
        torch.LongTensor(ids_val)
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    X_test, y_test, ids_test = data['test']
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test).unsqueeze(1),
        torch.LongTensor(y_test),
        torch.LongTensor(ids_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_id_loss = 0
    total_class_loss = 0
    id_preds = []
    id_targets = []
    class_preds = []
    class_targets = []
    
    for _, (data, y_class, y_id) in enumerate(train_loader):
        data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
        
        optimizer.zero_grad()
        id_logits, class_logits, _ = model(data)

        id_loss = criterion(id_logits, y_id) 
        class_loss = criterion(class_logits, y_class) 

        total_loss = id_loss + class_loss
        total_loss.backward()
        optimizer.step()
     
        total_id_loss += id_loss.item()
        total_class_loss += class_loss.item()

        _, id_pred = torch.max(id_logits, 1)
        _, class_pred = torch.max(class_logits, 1)
        
        id_preds.extend(id_pred.cpu().numpy())
        id_targets.extend(y_id.cpu().numpy())
        class_preds.extend(class_pred.cpu().numpy())
        class_targets.extend(y_class.cpu().numpy())

    id_accuracy = accuracy_score(id_targets, id_preds)
    class_accuracy = accuracy_score(class_targets, class_preds)
        
    return {
        'id_loss': total_id_loss / len(train_loader),
        'class_loss': total_class_loss / len(train_loader),
        'id_accuracy': id_accuracy,
        'class_accuracy': class_accuracy
    }

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_id_loss = 0
    total_class_loss = 0
    id_preds = []
    id_targets = []
    class_preds = []
    class_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            id_logits, class_logits, _ = model(data)

            id_loss = criterion(id_logits, y_id)
            class_loss = criterion(class_logits, y_class)

            total_id_loss += id_loss.item()
            total_class_loss += class_loss.item()
            
            _, id_pred = torch.max(id_logits, 1)
            _, class_pred = torch.max(class_logits, 1)
            
            id_preds.extend(id_pred.cpu().numpy())
            id_targets.extend(y_id.cpu().numpy())
            class_preds.extend(class_pred.cpu().numpy())
            class_targets.extend(y_class.cpu().numpy())
    
    id_accuracy = accuracy_score(id_targets, id_preds)
    class_accuracy = accuracy_score(class_targets, class_preds)
    
    return {
        'id_loss': total_id_loss / len(data_loader),
        'class_loss': total_class_loss / len(data_loader),
        'id_accuracy': id_accuracy,
        'class_accuracy': class_accuracy,
    }


if __name__ == "__main__":
    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
    data = data_loader.preprocess_data(random_state=RANDOM_STATE)
    train_loader, val_loader, test_loader = prepare_dataloaders(data)
    
    model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_id_losses = []
    train_class_losses = []
    train_id_accs = []
    train_class_accs = []
    val_id_losses = []
    val_class_losses = []
    val_id_accs = []
    val_class_accs = []

    train_metrics = evaluate(model, train_loader, criterion, DEVICE)
    train_id_losses.append(train_metrics['id_loss'])
    train_class_losses.append(train_metrics['class_loss'])
    train_id_accs.append(train_metrics['id_accuracy'])
    train_class_accs.append(train_metrics['class_accuracy'])
    val_metrics = evaluate(model, val_loader, criterion, DEVICE)
    val_id_losses.append(val_metrics['id_loss'])
    val_class_losses.append(val_metrics['class_loss'])
    val_id_accs.append(val_metrics['id_accuracy'])
    val_class_accs.append(val_metrics['class_accuracy'])
    
    # Train loop
    start_time = time.time()
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_id_losses.append(train_metrics['id_loss'])
        train_class_losses.append(train_metrics['class_loss'])
        train_id_accs.append(train_metrics['id_accuracy'])
        train_class_accs.append(train_metrics['class_accuracy'])

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        val_id_losses.append(val_metrics['id_loss'])
        val_class_losses.append(val_metrics['class_loss'])
        val_id_accs.append(val_metrics['id_accuracy'])
        val_class_accs.append(val_metrics['class_accuracy'])

        print(f'Train - ID Loss: {train_metrics["id_loss"]:.4f}, Class Loss: {train_metrics["class_loss"]}')
        print(f'Val - ID Acc: {val_metrics["id_accuracy"]:.4f}, Class Acc: {val_metrics["class_accuracy"]:.4f}')
        
        # Save best model
        if val_metrics['class_accuracy'] + val_metrics['id_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['class_accuracy'] + val_metrics['id_accuracy']
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, BEST))
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time(sec): {training_time:.2f}") 

    # Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, FINAL))

    np.savez(os.path.join(RESULT_PATH, f'{RANDOM_STATE}_ican_training_results.npz'),
         train_id_losses=train_id_losses,
         train_class_losses=train_class_losses,
         train_id_accs=train_id_accs,
         train_class_accs=train_class_accs,
         val_id_losses=val_id_losses,
         val_class_losses=val_class_losses,
         val_id_accs=val_id_accs,
         val_class_accs=val_class_accs)
    