import os
import wfdb

path = "data/mitdb/"
patient_id = wfdb.get_record_list('mitdb')
label = {}

for id in patient_id:
    record = wfdb.rdsamp(os.path.join(path, id))
    annotation = wfdb.rdann(os.path.join(path, id), 'atr')

    symbols = annotation.symbol

    for s in symbols:
        if s not in label:
            label[s] = 1
        else:
            label[s] += 1

classes = {'N': ['N', 'L', 'R', 'e', 'j'], 
           'S': ['A', 'a', 'J', 'S'], 
           'V': ['V', 'E'], 
           'F': ['F'], 
           'Q': ['/', 'f', 'Q']}

for c in classes:
    total = 0
    for i in classes[c]:
        print(i, label[i])
        total += label[i] 
    print(c, "Total", total)
    print()
