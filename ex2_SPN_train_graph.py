import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RESULT_PATH = 'results'
RANDOM_STATES = [101, 157, 223, 399, 482, 590, 696, 777, 829, 999]


if __name__ == "__main__":
    all_train_id_losses = []
    all_train_class_losses = []
    all_train_id_accs = []
    all_train_class_accs = []
    all_val_id_losses = []
    all_val_class_losses = []
    all_val_id_accs = []
    all_val_class_accs = []

    for RANDOM_STATE in RANDOM_STATES:
        data = np.load(os.path.join(RESULT_PATH, f'{RANDOM_STATE}_spn_training_results.npz'))

        all_train_id_losses.append(data['train_id_losses'])
        all_train_class_losses.append(data['train_class_losses'])
        all_train_id_accs.append(data['train_id_accs'])
        all_train_class_accs.append(data['train_class_accs'])
        all_val_id_losses.append(data['val_id_losses'])
        all_val_class_losses.append(data['val_class_losses'])
        all_val_id_accs.append(data['val_id_accs'])
        all_val_class_accs.append(data['val_class_accs'])

    all_train_id_losses = np.array(all_train_id_losses)
    all_train_class_losses = np.array(all_train_class_losses)
    all_train_id_accs = np.array(all_train_id_accs)
    all_train_class_accs = np.array(all_train_class_accs)
    all_val_id_losses = np.array(all_val_id_losses)
    all_val_class_losses = np.array(all_val_class_losses)
    all_val_id_accs = np.array(all_val_id_accs)
    all_val_class_accs = np.array(all_val_class_accs)

    mean_train_id_losses = np.mean(all_train_id_losses, axis=0)
    mean_train_class_losses = np.mean(all_train_class_losses, axis=0)
    mean_train_id_accs = np.mean(all_train_id_accs, axis=0)
    mean_train_class_accs = np.mean(all_train_class_accs, axis=0)
    mean_val_id_losses = np.mean(all_val_id_losses, axis=0)
    mean_val_class_losses = np.mean(all_val_class_losses, axis=0)
    mean_val_id_accs = np.mean(all_val_id_accs, axis=0)
    mean_val_class_accs = np.mean(all_val_class_accs, axis=0)
    
    epochs = range(0, len(mean_train_id_accs))

    matplotlib.rcParams['font.size'] = 15

    plt.figure(figsize=(7, 5))
    plt.xticks(range(0, 51, 10))
    plt.yticks(range(0, 101, 10))
    plt.xlim(-1, 51)
    plt.ylim(-1, 101)
    plt.plot(epochs, mean_train_id_accs * 100, 'r-', label='Identification Training Accuracy')
    plt.plot(epochs, mean_val_id_accs * 100, 'y-o', markevery=5, label='Identification Validation Accuracy')
    plt.plot(epochs, mean_train_class_accs * 100, 'b--', label='Classification Training Accuracy')
    plt.plot(epochs, mean_val_class_accs * 100, 'g--^', markevery=5, label='Classification Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig7a.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(7, 5))
    plt.xticks(range(0, 51, 10))
    plt.xlim(-1, 51)
    plt.ylim(0, 0.25)
    plt.plot(epochs, mean_train_id_losses, 'r-', label='Identification Training Loss')
    plt.plot(epochs, mean_val_id_losses, 'y-o', markevery=5, label='Identification Validation Loss')
    plt.plot(epochs, mean_train_class_losses, 'b--', label='Classification Training Loss')
    plt.plot(epochs, mean_val_class_losses, 'g--^', markevery=5, label='Classification Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig7b.png'), dpi=600, bbox_inches='tight')
    plt.show()
