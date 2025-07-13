import os
import wfdb
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=3):
    # Calculate Nyquist frequency
    nyquist = 0.5 * fs
    
    # Normalized cutoff frequencies
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Apply filter
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

class ECGDataLoader:
    def __init__(self, data_path="data/mitdb/", segment_length=1800):
        self.data_path = data_path
        self.segment_length = segment_length
        
        # AAMI EC57 standard class mapping
        self.aami_mapping = {
            'N': ['N', 'L', 'R', 'e', 'j'],
            'S': ['A', 'a', 'J', 'S'],
            'V': ['V', 'E'],
            'F': ['F'],
            'Q': ['/', 'f', 'Q']
        }
    
    def map_label_to_class(self, label):
        for class_name, labels in self.aami_mapping.items():
            if label in labels:
                return class_name
            
        return 'X'  # Not using this beat type
    
    def segment_ecg(self, signal, r_peaks, classes):
        """Segment ECG signal based on R-peaks"""          
        segments = []
        segment_classes = []
        half = self.segment_length // 2
        
        for i, peak in enumerate(r_peaks):
            if peak - half >= 0 and peak + half < len(signal):
                seg_start = peak - half
                seg_end = peak + half

                seg_classes = set([classes[i]]) # Center class

                j = i - 1
                while j >= 0 and r_peaks[j] >= seg_start:
                    seg_classes.add(classes[j])
                    j -= 1

                j = i + 1
                while j < len(r_peaks) and r_peaks[j] < seg_end:
                    seg_classes.add(classes[j])
                    j += 1

                if 'X' in seg_classes:
                    continue
                elif len(seg_classes) == 1:
                    segment_class = list(seg_classes)[0]
                elif len(seg_classes) == 2 and 'N' in seg_classes:
                    seg_classes.remove('N')
                    segment_class = list(seg_classes)[0]
                else:
                    continue

                segment = signal[peak - half:peak + half]
                segments.append(segment)
                segment_classes.append(segment_class)
                
        return np.array(segments), np.array(segment_classes)
    
    def preprocess_data(self, patient_ids=None, test_num=100, val_size=0.1, random_state=42):
        """Preprocess the ECG data and split into train, validation, and test sets"""
        if patient_ids is None:
            patient_ids = wfdb.get_record_list('mitdb')

        id_dic = {}
        for i in range(len(patient_ids)):
            id_dic[patient_ids[i]] = i

        all_segments = []
        all_classes = []
        all_ids = []
        
        for id in patient_ids:
            record = wfdb.rdrecord(os.path.join(self.data_path, id))
            annotation = wfdb.rdann(os.path.join(self.data_path, id), 'atr')
            
            signal = record.p_signal[:, 0]
            signal = butterworth_bandpass_filter(signal, lowcut=0.6, highcut=40, fs=record.fs, order=3)
            r_peaks = annotation.sample
            labels = annotation.symbol
            
            # Map labels to AAMI classes
            aami_classes = [self.map_label_to_class(label) for label in labels]
            
            # Segment the ECG
            segments, segment_classess = self.segment_ecg(signal, r_peaks, aami_classes)

            all_segments.extend(segments)
            all_classes.extend(segment_classess)
            all_ids.extend([id_dic[id]] * len(segments))
                
        X = np.array(all_segments)
        y = np.array(all_classes)
        ids = np.array(all_ids)
        
        classes = ['N', 'S', 'V', 'F', 'Q']
        y_idx = np.array([classes.index(label) for label in y])
        
        X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = [], [], [], [], [], []
        
        for cls in range(len(classes)):
            cls_indices = np.where(y_idx == cls)[0]
            
            np.random.seed(random_state)
            test_indices = np.random.choice(cls_indices, test_num, replace=False)
            train_val_indices = np.array([i for i in cls_indices if i not in test_indices])
                
            X_test.append(X[test_indices])
            y_test.append(y_idx[test_indices])
            ids_test.append(ids[test_indices])
            
            X_train_val.append(X[train_val_indices])
            y_train_val.append(y_idx[train_val_indices])
            ids_train_val.append(ids[train_val_indices])
            
        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)
        ids_test = np.concatenate(ids_test)
           
        X_train_val = np.vstack(X_train_val)
        y_train_val = np.concatenate(y_train_val)
        ids_train_val = np.concatenate(ids_train_val)
        
        # Split into training and validation
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_train_val, y_train_val, ids_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_val
        )
                
        return {
            'train': (X_train, y_train, ids_train),
            'val': (X_val, y_val, ids_val),
            'test': (X_test, y_test, ids_test),
        }
        