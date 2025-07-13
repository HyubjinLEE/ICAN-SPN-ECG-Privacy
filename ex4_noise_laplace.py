import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
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

def evaluate(model, data_loader, device, laplace_scale):
    model.eval()
    id_preds = []
    id_targets = []
    class_preds = []
    class_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            features = model.feature_extractor(data)

            # Add Laplace Noise
            if laplace_scale == 0:
                perturbed_features = features
            else:
                noise = torch.distributions.Laplace(0, laplace_scale).sample(features.shape).to(features.device)
                perturbed_features = noise + features
            
            id_logits = model.identification_module(perturbed_features)
            class_logits = model.classification_module(perturbed_features)
            
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
    # By ex2_ICAN_test_acc.py
    mean_ICAN_id_acc = 99.70
    mean_ICAN_class_acc = 95.60

    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
    model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)

    laplace_scales = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    all_id_accs = {scale: [] for scale in laplace_scales}
    all_class_accs = {scale: [] for scale in laplace_scales}

    for RANDOM_STATE in RANDOM_STATES:
        MODEL = str(RANDOM_STATE) + '_ican_final.pth'
        data = data_loader.preprocess_data(random_state=RANDOM_STATE)
        test_loader = prepare_dataloaders(data)

        # Load model
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, MODEL)))

        for laplace_scale in laplace_scales:
            test_metrics = evaluate(model, test_loader, DEVICE, laplace_scale=laplace_scale)
            all_id_accs[laplace_scale].append(test_metrics['id_accuracy'])
            all_class_accs[laplace_scale].append(test_metrics['class_accuracy'])

    # Calculate mean accuracies
    mean_id_accs = []
    mean_class_accs = []
    
    for laplace_scale in laplace_scales:
        mean_id_acc = sum(all_id_accs[laplace_scale]) / len(RANDOM_STATES) * 100
        mean_class_acc = sum(all_class_accs[laplace_scale]) / len(RANDOM_STATES) * 100
        
        mean_id_accs.append(mean_id_acc)
        mean_class_accs.append(mean_class_acc)
    
    matplotlib.rcParams['font.size'] = 15

    plt.figure(figsize=(10, 7))
    x_positions = np.arange(len(laplace_scales))  
    plt.xticks(x_positions, [str(s) for s in laplace_scales])  
    plt.yticks(range(0, 101, 10))
    plt.ylim(-1, 101)
    plt.axhline(y=mean_ICAN_id_acc, color='r', linestyle='-', label='Identification without Noise')
    plt.plot(x_positions, mean_id_accs, 'y-o', label='Identification with Noise')
    plt.axhline(y=mean_ICAN_class_acc, color='b', linestyle='--', label='Classification without Noise')
    plt.plot(x_positions, mean_class_accs, 'g--^', label='Classification with Noise')
    plt.xlabel('Lap(0, b)')
    plt.ylabel('Accuracy (%)')   
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig8a.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # Result
    print(f"{'Laplace Scale':<15} {'ID Acc':<15} {'Class Acc':<15}")
    print("-" * 45)
    for i, scale in enumerate(laplace_scales):
        print(f"{scale:<15.1f} {mean_id_accs[i]:<15.2f} {mean_class_accs[i]:<15.2f}")
        