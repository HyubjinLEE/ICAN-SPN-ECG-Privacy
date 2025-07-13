import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from model_ICAN import ICAN
from model_SPN import SPN
from data_loader import ECGDataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

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

def evaluate_model(model, data_loader, device, model_type='ican'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, y_class, y_id in data_loader:
            data, y_class, y_id = data.to(device), y_class.to(device), y_id.to(device)
            
            if model_type == 'ican':
                _, class_logits, _ = model(data)
            else:  # spn
                _, class_logits, _, _, _ = model(data)
            
            _, class_pred = torch.max(class_logits, 1)
            
            all_preds.extend(class_pred.cpu().numpy())
            all_targets.extend(y_class.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets)

def calculate_metrics(y_true, y_pred):
    # Overall accuracy
    all_accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    f1 = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(conf_matrix) - tp - fp - fn

        class_acc = (tp + tn) / (tp + tn + fp + fn)
        class_pre = tp / (tp + fp)
        class_sen = tp / (tp + fn)
        class_spe = tn / (tn + fp)
        class_f1 = 2 * (class_pre * class_sen) / (class_pre + class_sen)
        
        accuracy.append(class_acc)
        precision.append(class_pre)
        sensitivity.append(class_sen)
        specificity.append(class_spe)
        f1.append(class_f1)    

    accuracy = np.array(accuracy)
    precision = np.array(precision)
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)
    f1 = np.array(f1)
    
    # Macro average
    macro_accuracy = np.mean(accuracy)
    macro_precision = np.mean(precision)
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_f1 = np.mean(f1)
    
    return {
        'all_accuracy': all_accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'macro_accuracy' : macro_accuracy,
        'macro_precision': macro_precision,
        'macro_sensitivity': macro_sensitivity,
        'macro_specificity': macro_specificity,
        'macro_f1': macro_f1,
        'conf_matrix': conf_matrix
    }

def print_metrics(metrics, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    
    print(f"\nOverall Accuracy: {metrics['all_accuracy']*100:.2f}%")
    
    print(f"\nMacro Average:")
    print(f"  Accuracy: {metrics['macro_accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['macro_precision']*100:.2f}%")
    print(f"  Sensitivity: {metrics['macro_sensitivity']*100:.2f}%")
    print(f"  Specificity: {metrics['macro_specificity']*100:.2f}%")
    print(f"  F1-score: {metrics['macro_f1']*100:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Accuracy':<12} {'Precision':<12} {'Sensitivity':<12} {'Specificity':<12} {'F1-score':<12}")
    print("-" * 60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<10} "
              f"{metrics['accuracy'][i]*100:<12.2f} "
              f"{metrics['precision'][i]*100:<12.2f} "
              f"{metrics['sensitivity'][i]*100:<12.2f} "
              f"{metrics['specificity'][i]*100:<12.2f} "
              f"{metrics['f1'][i]*100:<12.2f}")

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 9))
        
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig10.png'), dpi=600, bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(ican_metrics, spn_metrics):
    """ICAN vs SPN"""
    
    # 1. Accuracy by class
    plt.figure(figsize=(10, 7))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    ican_acc = ican_metrics['accuracy'] * 100
    spn_acc = spn_metrics['accuracy'] * 100
    
    bars1 = plt.bar(x - width/2, ican_acc, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x + width/2, spn_acc, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, CLASS_NAMES)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12a.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 2. Precision by class
    plt.figure(figsize=(10, 7))
    
    ican_prec = ican_metrics['precision'] * 100
    spn_prec = spn_metrics['precision'] * 100
    
    bars1 = plt.bar(x - width/2, ican_prec, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x + width/2, spn_prec, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Precision (%)')
    plt.xticks(x, CLASS_NAMES)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12b.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 3. Sensitivity by class
    plt.figure(figsize=(10, 7))
    
    ican_sens = ican_metrics['sensitivity'] * 100
    spn_sens = spn_metrics['sensitivity'] * 100
    
    bars1 = plt.bar(x - width/2, ican_sens, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x + width/2, spn_sens, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Sensitivity (%)')
    plt.xticks(x, CLASS_NAMES)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12c.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 4. Specificity by class
    plt.figure(figsize=(10, 7))
    
    ican_spec = ican_metrics['specificity'] * 100
    spn_spec = spn_metrics['specificity'] * 100
    
    bars1 = plt.bar(x - width/2, ican_spec, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x + width/2, spn_spec, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Specificity (%)')
    plt.xticks(x, CLASS_NAMES)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12d.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 5. F1-Score by class
    plt.figure(figsize=(10, 7))
    
    ican_f1 = ican_metrics['f1'] * 100
    spn_f1 = spn_metrics['f1'] * 100
    
    bars1 = plt.bar(x - width/2, ican_f1, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x + width/2, spn_f1, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('F1-Score (%)')
    plt.xticks(x, CLASS_NAMES)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12e.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # 6. Macro Average by class
    plt.figure(figsize=(10, 7))
    
    metrics_names = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score']
    ican_macro = [
        ican_metrics['macro_accuracy'] * 100,
        ican_metrics['macro_precision'] * 100,
        ican_metrics['macro_sensitivity'] * 100,
        ican_metrics['macro_specificity'] * 100,
        ican_metrics['macro_f1'] * 100
    ]
    spn_macro = [
        spn_metrics['macro_accuracy'] * 100,
        spn_metrics['macro_precision'] * 100,
        spn_metrics['macro_sensitivity'] * 100,
        spn_metrics['macro_specificity'] * 100,
        spn_metrics['macro_f1'] * 100
    ]
    
    x_macro = np.arange(len(metrics_names))
    
    bars1 = plt.bar(x_macro - width/2, ican_macro, width, label='ICAN', color="skyblue")
    bars2 = plt.bar(x_macro + width/2, spn_macro, width, label='ICAN with SPN noise', color="lightcoral", hatch='//')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
    
    plt.ylabel('Average (%)')
    plt.xticks(x_macro, metrics_names)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig12f.png'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    data_loader = ECGDataLoader(data_path="data/mitdb/", segment_length=SEG_LENGTH*360)
    ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)
    
    all_ican_preds = []
    all_ican_targets = []
    all_spn_preds = []
    all_spn_targets = []
    
    for RANDOM_STATE in RANDOM_STATES:
        data = data_loader.preprocess_data(random_state=RANDOM_STATE)
        test_loader = prepare_dataloaders(data)
        
        # ICAN evaluate
        ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
        ican_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f'{RANDOM_STATE}_ican_final.pth')))
        
        ican_preds, ican_targets = evaluate_model(ican_model, test_loader, DEVICE, model_type='ican')
        all_ican_preds.extend(ican_preds)
        all_ican_targets.extend(ican_targets)
        
        # SPN evaluate
        spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)
        spn_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f'{RANDOM_STATE}_spn_best.pth')))
        
        spn_preds, spn_targets = evaluate_model(spn_model, test_loader, DEVICE, model_type='spn')
        all_spn_preds.extend(spn_preds)
        all_spn_targets.extend(spn_targets)
    
    all_ican_preds = np.array(all_ican_preds)
    all_ican_targets = np.array(all_ican_targets)
    all_spn_preds = np.array(all_spn_preds)
    all_spn_targets = np.array(all_spn_targets)
    
    matplotlib.rcParams['font.size'] = 15

    # ICAN 
    ican_metrics = calculate_metrics(all_ican_targets, all_ican_preds)
    print_metrics(ican_metrics, "ICAN")
    
    # SPN 
    spn_metrics = calculate_metrics(all_spn_targets, all_spn_preds)
    print_metrics(spn_metrics, "SPN")
    plot_confusion_matrix(spn_metrics['conf_matrix'])
    
    # ICAN vs SPN
    plot_metrics_comparison(ican_metrics, spn_metrics)
