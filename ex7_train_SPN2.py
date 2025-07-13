import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from model_SPN import SPN
from ex7_data_loader2 import ECGDataLoader
from sklearn.metrics import accuracy_score

NUM_CLASSES = 3
NUM_PATIENTS = 48
MODEL_SAVE_PATH = 'saved_models2'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
SEG_LENGTH = 360*5
RANDOM_STATE = 300
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
FEATURE_DIM = 256
D_K = 64

ICAN_LOAD = str(RANDOM_STATE) + '_ican_final.pth'
FINAL = str(RANDOM_STATE) + '_spn_final.pth'
BEST = str(RANDOM_STATE) + '_spn_best.pth'

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

def train_epoch(model, train_loader, optimizer, id_criterion, class_criterion, device):
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
        id_logits, class_logits, _, _, _ = model(data)
        log_probs = F.log_softmax(id_logits, dim=1)
        uniform_target = torch.full_like(log_probs, 1.0 / NUM_PATIENTS) 
        id_loss = id_criterion(log_probs, uniform_target)
        id_loss.backward()
        optimizer.step()
        total_id_loss += id_loss.item()
    
        optimizer.zero_grad()
        id_logits, class_logits, _, _, _ = model(data)
        class_loss = class_criterion(class_logits, y_class)
        class_loss.backward()
        optimizer.step()
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

def evaluate(model, data_loader, id_criterion, class_criterion, device):
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
            
            id_logits, class_logits, _, _, _ = model(data)

            log_probs = F.log_softmax(id_logits, dim=1)
            uniform_target = torch.full_like(log_probs, 1.0 / NUM_PATIENTS) 
            id_loss = id_criterion(log_probs, uniform_target)
            class_loss = class_criterion(class_logits, y_class)

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
    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH)
    data = data_loader.preprocess_data(random_state=RANDOM_STATE)
    train_loader, val_loader, test_loader = prepare_dataloaders(data)
    
    # ICAN load
    ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    ican_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, ICAN_LOAD)))
    
    # SPN model initialization
    spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)
    id_criterion = nn.KLDivLoss(reduction='batchmean')
    class_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(spn_model.parameters(), lr=LEARNING_RATE)

    train_id_losses = []
    train_class_losses = []
    train_id_accs = []
    train_class_accs = []
    val_id_losses = []
    val_class_losses = []
    val_id_accs = []
    val_class_accs = []
    test_id_losses = []
    test_class_losses = []
    test_id_accs = []
    test_class_accs = []
    
    train_metrics = evaluate(spn_model, train_loader, id_criterion, class_criterion, DEVICE)
    train_id_losses.append(train_metrics['id_loss'])
    train_class_losses.append(train_metrics['class_loss'])
    train_id_accs.append(train_metrics['id_accuracy'])
    train_class_accs.append(train_metrics['class_accuracy'])
    val_metrics = evaluate(spn_model, val_loader, id_criterion, class_criterion, DEVICE)
    val_id_losses.append(val_metrics['id_loss'])
    val_class_losses.append(val_metrics['class_loss'])
    val_id_accs.append(val_metrics['id_accuracy'])
    val_class_accs.append(val_metrics['class_accuracy'])
    test_metrics = evaluate(spn_model, test_loader, id_criterion, class_criterion, DEVICE)
    test_id_losses.append(test_metrics['id_loss'])
    test_class_losses.append(test_metrics['class_loss'])
    test_id_accs.append(test_metrics['id_accuracy'])
    test_class_accs.append(test_metrics['class_accuracy'])

    # Train loop
    start_time = time.time()
    best_val_metric = 0.0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        # Train
        train_metrics = train_epoch(spn_model, train_loader, optimizer, id_criterion, class_criterion, DEVICE)
        train_id_losses.append(train_metrics['id_loss'])
        train_class_losses.append(train_metrics['class_loss'])
        train_id_accs.append(train_metrics['id_accuracy'])
        train_class_accs.append(train_metrics['class_accuracy'])

        # Validation
        val_metrics = evaluate(spn_model, val_loader, id_criterion, class_criterion, DEVICE)
        val_id_losses.append(val_metrics['id_loss'])
        val_class_losses.append(val_metrics['class_loss'])
        val_id_accs.append(val_metrics['id_accuracy'])
        val_class_accs.append(val_metrics['class_accuracy'])

        # Test
        test_metrics = evaluate(spn_model, test_loader, id_criterion, class_criterion, DEVICE)
        test_id_losses.append(test_metrics['id_loss'])
        test_class_losses.append(test_metrics['class_loss'])
        test_id_accs.append(test_metrics['id_accuracy'])
        test_class_accs.append(test_metrics['class_accuracy'])

        print(f'Train - ID Loss: {train_metrics["id_loss"]:.4f}, Class Loss: {train_metrics["class_loss"]}')
        print(f'Val - ID Acc: {val_metrics["id_accuracy"]:.4f}, Class Acc: {val_metrics["class_accuracy"]:.4f}')
        print(f'Test - ID Acc: {test_metrics["id_accuracy"]:.4f}, Class Acc: {test_metrics["class_accuracy"]:.4f}')
        
        # Save best model
        val_metric = val_metrics['id_accuracy'] - val_metrics['class_accuracy']
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save(spn_model.state_dict(), os.path.join(MODEL_SAVE_PATH, BEST))

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time(sec): {training_time:.2f}") 
