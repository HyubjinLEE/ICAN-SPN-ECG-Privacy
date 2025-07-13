import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from model_SPN import SPN
from data_loader import ECGDataLoader


NUM_CLASSES = 5
NUM_PATIENTS = 48
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
MODEL_SAVE_PATH = 'saved_models'
RESULT_PATH = 'results'
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

def evaluate(model, data_loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            _, class_logits, _, _, _ = model(data)
            
            class_probs = torch.softmax(class_logits, dim=1)
            
            all_probs.extend(class_probs.cpu().numpy())
            all_targets.extend(y_class.cpu().numpy())
    
    return np.array(all_probs), np.array(all_targets)

def calculate_roc_auc(y_true, y_probs):
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro Average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc

def plot_roc_curves(fpr, tpr, roc_auc):
    matplotlib.rcParams['font.size'] = 15
    plt.figure(figsize=(8, 8))
    
    colors = cycle(['blue', 'orange', 'yellow', 'red', 'green'])
    linestyles = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1))])  

    # ROC curves for each class
    for i, (color, linestyle) in zip(range(NUM_CLASSES), zip(colors, linestyles)):
        plt.plot(fpr[i], tpr[i], color=color, linestyle=linestyle, label=f'Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Macro Average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], color='magenta', linestyle=(0, (1, 1)), label=f'Average (AUC = {roc_auc["macro"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULT_PATH, 'fig11.png'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
    ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)   

    all_probs = []
    all_targets = []
    
    for RANDOM_STATE in RANDOM_STATES:
        data = data_loader.preprocess_data(random_state=RANDOM_STATE)
        test_loader = prepare_dataloaders(data)
        
        # Model load
        spn_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f'{RANDOM_STATE}_spn_best.pth')))
        
        probs, targets = evaluate(spn_model, test_loader, DEVICE)
        all_probs.extend(probs)
        all_targets.extend(targets)
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Calculate ROC curve
    fpr, tpr, roc_auc = calculate_roc_auc(all_targets, all_probs)

    plot_roc_curves(fpr, tpr, roc_auc)
