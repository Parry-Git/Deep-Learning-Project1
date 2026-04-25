import mynn as nn
import numpy as np
from struct import unpack
import gzip
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = nn.models.Model_MLP()
model.load_model(BASE_DIR / 'best_models' / 'best_model.pickle')

test_images_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-images-idx3-ubyte.gz'
test_labels_path = BASE_DIR / 'dataset' / 'MNIST' / 't10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28).astype(np.float32)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255.0

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))
