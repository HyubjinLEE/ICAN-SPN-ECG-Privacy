import os
import wfdb

# MIT-BIH Arrhythmia Database download 
if not os.path.exists("data/mitdb"):
    os.makedirs("data", exist_ok=True)
    wfdb.dl_database("mitdb", "data/mitdb")
