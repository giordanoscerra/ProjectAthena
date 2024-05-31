import os
import sys

import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from scoring import SCHOOLS
from utilities import getData

# Load data
tr, vl, _ = getData(min_chars=20, max_chars=1700)
batchSize = 38
num_epochs = 3
print(f'training quantity: {len(tr)}', f'validation quantity: {len(vl)}')
texts_tr = tr['sentence_str']
labels_tr = tr['school']
texts_vl = vl['sentence_str']
labels_vl = vl['school']

labels_tr = torch.tensor([SCHOOLS.index(label) for label in labels_tr])
labels_vl = torch.tensor([SCHOOLS.index(label) for label in labels_vl])
