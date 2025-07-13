import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RESULT_PATH = 'results'
RANDOM_STATES = [101, 157, 223, 399, 482, 590, 696, 777, 829, 999]


if __name__ == "__main__":
    # By ex2_ICAN_test_acc.py
    mean_ICAN_id_acc = 99.70
    mean_ICAN_class_acc = 95.60

    all_test_id_losses = []
    all_test_class_losses = []
    all_test_id_accs = []
    all_test_class_accs = []

    for RANDOM_STATE in RANDOM_STATES:
        data = np.load(os.path.join(RESULT_PATH, f'{RANDOM_STATE}_spn_training_results.npz'))

        all_test_id_losses.append(data['test_id_losses'])
        all_test_class_losses.append(data['test_class_losses'])
        all_test_id_accs.append(data['test_id_accs'])
        all_test_class_accs.append(data['test_class_accs'])

    all_test_id_losses = np.array(all_test_id_losses)
    all_test_class_losses = np.array(all_test_class_losses)
    all_test_id_accs = np.array(all_test_id_accs)
    all_test_class_accs = np.array(all_test_class_accs)

    mean_test_id_losses = np.mean(all_test_id_losses, axis=0)
    mean_test_class_losses = np.mean(all_test_class_losses, axis=0)
    mean_test_id_accs = np.mean(all_test_id_accs, axis=0)
    mean_test_class_accs = np.mean(all_test_class_accs, axis=0)

    epochs = range(0, len(mean_test_id_accs))

    matplotlib.rcParams['font.size'] = 15

    plt.figure(figsize=(10, 7))
    plt.xticks(range(0, 51, 10))
    plt.yticks(range(0, 101, 10))
    plt.xlim(-1, 51)
    plt.ylim(-1, 101)
    plt.axhline(y=mean_ICAN_id_acc, color='r', linestyle='-', label='Identification without Noise')
    plt.plot(epochs, mean_test_id_accs * 100, 'y-o', markevery=5, label='Identification with Noise')
    plt.axhline(y=mean_ICAN_class_acc, color='b', linestyle='--', label='Classification without Noise')
    plt.plot(epochs, mean_test_class_accs * 100, 'g--^', markevery=5, label='Classification with Noise')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'fig8b.png'), dpi=600, bbox_inches='tight')
    plt.show()
