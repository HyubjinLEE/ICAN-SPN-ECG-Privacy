import os
import time
import torch
import numpy as np
from thop import profile
from model_ICAN import ICAN
from model_SPN import SPN

NUM_CLASSES = 5
NUM_PATIENTS = 48
MODEL_SAVE_PATH = 'saved_models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
RANDOM_STATE = 101 
SEQ_LENGTH = 5
FEATURE_DIM = 256
D_K = 64

def measure_inference_time(model, model_name):
    times = []
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(1000):
            dummy_input = torch.randn(1, 1, SEQ_LENGTH*360).to(DEVICE)
            start_time = time.time()
            _ = model(dummy_input)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
            
    avg_time_ms = np.mean(times) * 1000
    print(f"{model_name} Average Inference Time: {avg_time_ms:.4f} ms")


def measure_flops_params(model, model_name):
    dummy_input = torch.randn(1, 1, SEQ_LENGTH*360).to(DEVICE)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
 
    flops = macs * 2
    print(f"{model_name} Parameters: {params / 1e6:.4f} M")
    print(f"{model_name} FLOPs: {flops / 1e9:.4f} G")

if __name__ == "__main__":
    # ICAN load
    ican_model = ICAN(feature_dim=FEATURE_DIM, d_k=D_K, num_patients=NUM_PATIENTS, num_classes=NUM_CLASSES).to(DEVICE)
    ican_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f'{RANDOM_STATE}_ican_final.pth')))
    ican_model.eval()

    # SPN load
    spn_model = SPN(ican_model, feature_dim=FEATURE_DIM).to(DEVICE)
    spn_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f'{RANDOM_STATE}_spn_best.pth')))
    spn_model.eval()

    print("Measuring ICAN Model...")
    measure_inference_time(ican_model, "ICAN")
    measure_flops_params(ican_model, "ICAN")
    
    print("-" * 30)

    print("Measuring SPN Model...")
    measure_inference_time(spn_model, "SPN")
    measure_flops_params(spn_model, "SPN")
    