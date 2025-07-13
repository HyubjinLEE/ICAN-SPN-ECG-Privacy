import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from data_loader import ECGDataLoader
from sklearn.metrics import accuracy_score

NUM_CLASSES = 5
NUM_PATIENTS = 48
MODEL_SAVE_PATH = 'saved_models/length'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
RANDOM_STATES = [399, 482, 590, 777, 999]
SEG_LENGTHS = [1, 2, 3, 5, 10]
BATCH_SIZE = 128
FEATURE_DIM = 256
D_K = 64

def prepare_dataloaders(data):
    X_test, y_test, ids_test = data['test']
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test).unsqueeze(1),
        torch.LongTensor(y_test),
        torch.LongTensor(ids_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return test_loader

def evaluate(model, data_loader, device):
    model.eval()
    id_preds = []
    id_targets = []
    class_preds = []
    class_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            id_logits, class_logits, _ = model(data)
            
            _, id_pred = torch.max(id_logits, 1)
            _, class_pred = torch.max(class_logits, 1)
            
            id_preds.extend(id_pred.cpu().numpy())
            id_targets.extend(y_id.cpu().numpy())
            class_preds.extend(class_pred.cpu().numpy())
            class_targets.extend(y_class.cpu().numpy())
    
    id_accuracy = accuracy_score(id_targets, id_preds)
    class_accuracy = accuracy_score(class_targets, class_preds)
    
    return {
        'id_accuracy': id_accuracy,
        'class_accuracy': class_accuracy,
    }


if __name__ == "__main__":
    for SEG_LENGTH in SEG_LENGTHS:
        data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
        model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)

        id_accs = []
        class_accs = []

        for RANDOM_STATE in RANDOM_STATES:
            MODEL = str(RANDOM_STATE) + '_ican_final_' + str(SEG_LENGTH) + '.pth'
            data = data_loader.preprocess_data(random_state=RANDOM_STATE)
            test_loader = prepare_dataloaders(data)

            # Load model
            model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, MODEL)))

            # Evaluate
            test_metrics = evaluate(model, test_loader, DEVICE)
            id_accs.append(test_metrics['id_accuracy'])
            class_accs.append(test_metrics['class_accuracy'])

        print("Segment Length:", SEG_LENGTH)
        print(f"  Identification Accuracy: {sum(id_accs)*100 / len(RANDOM_STATES):.2f}")
        print(f"  Classification Accuracy: {sum(class_accs)*100 / len(RANDOM_STATES):.2f}")
