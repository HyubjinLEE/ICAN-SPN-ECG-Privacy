import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from model_SPN import SPN
from data_loader import ECGDataLoader
from sklearn.metrics import accuracy_score

NUM_CLASSES = 5
NUM_PATIENTS = 48
RESULT_PATH = 'results'
MODEL_SAVE_PATH = 'saved_models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter
RANDOM_STATES = [101, 157, 223, 399, 482, 590, 696, 777, 829, 999]
SEG_LENGTH = 5
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

def evaluate(spn_model, data_loader, device, noise_factor):
    spn_model.eval()
    id_preds = []
    id_targets = []
    class_preds = []
    class_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            _, _, features, noise, _ = spn_model(data)
            
            # Adjust noise strength
            noise = noise * noise_factor
            perturbed_features = features + noise
            perturbed_features = perturbed_features.unsqueeze(-1)
            
            id_logits = spn_model.ican.identification_module(perturbed_features)
            class_logits = spn_model.ican.classification_module(perturbed_features)
            
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
    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
    ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)

    noise_factors = np.arange(0.0, 2.1, 0.1)

    all_id_accs = {factor: [] for factor in noise_factors}
    all_class_accs = {factor: [] for factor in noise_factors}
    
    for RANDOM_STATE in RANDOM_STATES:        
        MODEL = str(RANDOM_STATE) + '_spn_best.pth'
        data = data_loader.preprocess_data(random_state=RANDOM_STATE)
        test_loader = prepare_dataloaders(data)
        
        # Load SPN model
        spn_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, MODEL)))
        
        for noise_factor in noise_factors:
            test_metrics = evaluate(spn_model, test_loader, DEVICE, noise_factor=noise_factor)
            all_id_accs[noise_factor].append(test_metrics['id_accuracy'])
            all_class_accs[noise_factor].append(test_metrics['class_accuracy'])
    
    # Calculate mean accuracies
    mean_id_accs = []
    mean_class_accs = []
    
    for noise_factor in noise_factors:
        mean_id_acc = sum(all_id_accs[noise_factor]) / len(RANDOM_STATES) * 100
        mean_class_acc = sum(all_class_accs[noise_factor]) / len(RANDOM_STATES) * 100
        
        mean_id_accs.append(mean_id_acc)
        mean_class_accs.append(mean_class_acc)
    
    matplotlib.rcParams['font.size'] = 15

    plt.figure(figsize=(10, 7))
    plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    plt.yticks(range(0, 101, 10))
    plt.xlim(-0.01, 2.01)
    plt.ylim(-1, 101)
    plt.plot(noise_factors, mean_id_accs, 'r-', label='Identification Accuracy')
    plt.plot(noise_factors, mean_class_accs, 'b--', label='Classification Accuracy')
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Noise Strength')
    plt.ylabel('Accuracy')   
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig13.png'), dpi=600, bbox_inches='tight')
    plt.show()
