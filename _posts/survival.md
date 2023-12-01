```python
!df -h
```

    Filesystem      Size  Used Avail Use% Mounted on
    overlay         864G  727G   94G  89% /
    tmpfs            64M     0   64M   0% /dev
    tmpfs            63G     0   63G   0% /sys/fs/cgroup
    /dev/nvme0n1p4  864G  727G   94G  89% /workspace/home
    /dev/sdc1        11T  1.4T  9.0T  13% /workspace/data
    tmpfs            63G  133M   63G   1% /dev/shm
    tmpfs            63G   12K   63G   1% /proc/driver/nvidia
    tmpfs            13G  4.4M   13G   1% /run/nvidia-persistenced/socket
    udev             63G     0   63G   0% /dev/nvidia0
    tmpfs            63G     0   63G   0% /proc/asound
    tmpfs            63G     0   63G   0% /proc/acpi
    tmpfs            63G     0   63G   0% /proc/scsi
    tmpfs            63G     0   63G   0% /sys/firmware



```python
!nvidia-smi
```

    Thu Nov  9 08:03:27 2023       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.39       Driver Version: 460.39       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Quadro RTX 8000     Off  | 00000000:18:00.0 Off |                    0 |
    | N/A   41C    P0    61W / 250W |   4947MiB / 45556MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  Quadro RTX 8000     Off  | 00000000:86:00.0 Off |                    0 |
    | N/A   79C    P0   219W / 250W |   4592MiB / 45556MiB |     60%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    +-----------------------------------------------------------------------------+



```python
import csv
import os
import gc
import glob
import warnings
import random
import easydict
import copy
from collections import defaultdict
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm.notebook import tqdm
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split

import timm
import libtiff
import sys

import warnings

warnings.filterwarnings(action='ignore')
Image.MAX_IMAGE_PIXELS = None
libtiff.libtiff_ctypes.suppress_warnings()
gc.collect()
torch.cuda.empty_cache()
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams['axes.unicode_minus'] = False
```


```python
SEED = 42
RAND_CNT = 0
IMG_SIZE = (256, 256) # width, height
DROP_RATE = 0.0
EPOCH = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5
PATIENCE = 5
PRINT_EVERY = 1000

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
TOPK = 10

GPU_IDX = 1
```


```python
def seed_everything(random_seed: int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    

```


```python
seed_everything(SEED)
```


```python
def get_augmentation():
    _transform = [
#         A.Rotate(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ColorJitter(),
    ]
    return A.Compose(_transform)

def get_preprocessing():
    _transform = [
        A.Resize(IMG_SIZE[1], IMG_SIZE[0]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    return A.Compose(_transform)
```


```python
class MILDataset(Dataset): # MIL-RNN
    def __init__(self, slide_list, label_list, augmentation=False, preprocessing=False):
        self.slide_list = slide_list
        self.label_list = label_list
        self.augmentation = get_augmentation() if augmentation else None
        self.preprocessing = get_preprocessing() if preprocessing else None
        
        patch_path_list = []
        slide_idx = []
        for i, slide_name in enumerate(tqdm(self.slide_list)):
#             patch_path = f'/workspace/data4/changwoo/SDP/patch/A100_data/{MAGNIFICATION}X/*/*/{slide_name}/*.png'
            slide_name = slide_name.split('/')[-1].split('.')[0]
#             patch_path = '../MIL/patches/' + slide_name + '/*.npy'
#             patch_path = '/workspace/data/MIL_patches/10/256_50/' + slide_name + '/*.png'
            patch_path = '/workspace/data/MIL_patches/20/256_0/' + slide_name + '/*.png'
#             patch_path = '/workspace/data/MIL_patches/20/512_25/' + slide_name + '/*.png'
#             patch_path = '/workspace/data/MIL_patches/40/512_0/' + slide_name + '/*.png'
#             patch_path = '/workspace/data/MIL_patches/10/512_75/' + slide_name + '/*.png'


            finded_patch = glob.glob(patch_path)
            patch_path_list.extend(finded_patch)
            slide_idx.extend([i] * len(finded_patch))
        
        self.patch_path_list = patch_path_list
        self.slide_idx = slide_idx
    
    def set_mode(self, mode):
        self.mode = mode
    
    def make_topk_data(self, idxs):
        self.t_data = [(self.slide_idx[idx], self.patch_path_list[idx], self.label_list[self.slide_idx[idx]]) for idx in idxs]
    
    def shuffle_topk_data(self, random_seed):
        random.seed(random_seed)
        self.t_data = random.sample(self.t_data, len(self.t_data))
    
    def __getitem__(self, idx):
        if self.mode == 1: # eval
            slide_idx = self.slide_idx[idx]
            img = cv2.imread(self.patch_path_list[idx])
#             img = np.load(self.patch_path_list[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.preprocessing:
                    sample = self.preprocessing(image=img)
                    img = sample['image']
            label = self.label_list[slide_idx]
            return img, label
        elif self.mode == 2: # train
            slide_idx, patch_path, label = self.t_data[idx]
            img = cv2.imread(patch_path)
#             img = np.load(self.patch_path_list[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.augmentation:
                    sample = self.augmentation(image=img)
                    img = sample['image']
            if self.preprocessing:
                    sample = self.preprocessing(image=img)
                    img = sample['image']
            return img, label
    
    def __len__(self):
        if self.mode == 1:
            return len(self.patch_path_list)
        elif self.mode == 2:
            return len(self.t_data)
```


```python
df = pd.read_csv('../fuhrman_grading/data/fuhrman_label.csv')

_train, test = train_test_split(df, test_size=0.2, stratify=df.fuhrman_nuclear_grade)
train, valid = train_test_split(_train, test_size=0.2, stratify=_train.fuhrman_nuclear_grade)

train_list, valid_list, test_list = train['path'].tolist(), valid['path'].tolist(), test['path'].tolist()
train_label, valid_label, test_label = train['fuhrman_nuclear_grade'].tolist(), valid['fuhrman_nuclear_grade'].tolist(), test['fuhrman_nuclear_grade'].tolist()
```


```python
# train_data = MILDataset(train_list, train_label, augmentation=True, preprocessing=True)
# valid_data = MILDataset(valid_list, valid_label, augmentation=False, preprocessing=True)

# train_data.set_mode(mode=1)
# valid_data.set_mode(mode=1)

# train_dataloader = DataLoader(train_data, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=8)
# valid_dataloader = DataLoader(valid_data, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=8)

# train_train_dataloader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)
# valid_valid_dataloader = DataLoader(valid_data, batch_size=TOPK, shuffle=False, num_workers=8)

test_data = MILDataset(test_list, test_label, augmentation=False, preprocessing=True)
test_data.set_mode(mode=1)
test_dataloader = DataLoader(test_data, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=8)
test_test_dataloader = DataLoader(test_data, batch_size=TOPK, shuffle=False, num_workers=8)
```


      0%|          | 0/439 [00:00<?, ?it/s]



```python
class ResNet_fc(nn.Module):
    def __init__(self, _backbone, num_classes=2):
        super().__init__()
        
        backbone = copy.deepcopy(_backbone)
        self.backbone = backbone
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
```


```python
backbone = timm.create_model('resnet50', pretrained=True, num_classes=1024, drop_rate=DROP_RATE)
model = ResNet_fc(copy.deepcopy(backbone), num_classes=2)
model.load_state_dict(torch.load('../weights/pancancer-based/10/512_75_lr5/best_epoch 0.pth', map_location='cpu'))
model = model.to(GPU_IDX)
criterion = nn.CrossEntropyLoss()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-49-61e22c2e9a3c> in <module>
          1 backbone = timm.create_model('resnet50', pretrained=True, num_classes=1024, drop_rate=DROP_RATE)
    ----> 2 model = ResNet_fc(copy.deepcopy(backbone), num_classes=2)
          3 model.load_state_dict(torch.load('../weights/pancancer-based/10/512_75_lr5/best_epoch 0.pth', map_location='cpu'))
          4 model = model.to(GPU_IDX)
          5 criterion = nn.CrossEntropyLoss()


    NameError: name 'ResNet_fc' is not defined



```python
seed_everything(SEED)

#################
#####  test #####
#################
print('#################')
print('#####  test #####')
print('#################')
test_data.set_mode(mode=1)
test_probs = model_inference(test_dataloader, model)
test_topk = group_argtopk(np.array(test_data.slide_idx), test_probs, TOPK)
test_data.make_topk_data(test_topk)

test_data.set_mode(mode=2)    
test_loss_list, test_label_list, test_prob_list, test_pred_list = model_valid(test_test_dataloader, model, criterion)
test_loss = np.mean(test_loss_list)
test_accuracy = accuracy_score(test_label_list, test_pred_list)
print(f'Loss: {np.mean(test_loss):.4f}, Acc: {test_accuracy:.4f}')
plot_confusion_matrix(confusion_matrix(test_label_list, test_pred_list), classes=['c0', 'c1'], title='test')
plt.show()
gc.collect()
torch.cuda.empty_cache()

result_df = {'test_label_list': test_label_list, 'test_prob_list': test_prob_list, 'test_pred_list':test_pred_list}
result = pd.DataFrame(result_df)
# DataFrame을 CSV 파일로 저장
result.to_csv('../logs/imagenet-based/256/result_10x_25_top25_lr5.csv', index=False)
```


```python
test_list_df = pd.DataFrame(test_list)
test_list_df.columns = ['path']
test_list_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>/workspace/data/parsed_level0/TA3322-C/TA3322-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>/workspace/data/parsed_level0/TA3323-A/TA3323-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>/workspace/data/parsed_level0/TA3323-B/TA3323-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>/workspace/data/parsed_level0/TA3290-C/TA3290-...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>/workspace/data/parsed_level0/TA3321-A/TA3321-...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>434</th>
      <td>/workspace/data/parsed_level0/TA3291-C/TA3291-...</td>
    </tr>
    <tr>
      <th>435</th>
      <td>/workspace/data/parsed_level0/TA3288-C/TA3288-...</td>
    </tr>
    <tr>
      <th>436</th>
      <td>/workspace/data/parsed_level0/TA2252-A/TA2252-...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>/workspace/data/parsed_level0/TA2257-B/TA2257-...</td>
    </tr>
    <tr>
      <th>438</th>
      <td>/workspace/data/parsed_level0/TA3291-C/TA3291-...</td>
    </tr>
  </tbody>
</table>
<p>439 rows × 1 columns</p>
</div>




```python
# result_df = pd.read_csv('./logs/imagenet-based/512/result_40x_0_top5_lr5.csv')
# result_df = pd.concat((result_df, test_list_df), axis=1)
# result_df
# result_df2 = pd.read_csv('./logs/pancancer-based/256/top10/result_20x_0_top10_lr5.csv')
# result_df2 = pd.read_csv('./logs/imagenet-based/256/top10/result_20x_0_top10_lr5.csv')
result_df2 = pd.read_csv('./logs/scratch-based/256/result_20x_0_top10_lr5.csv')


result_df2 = pd.concat((result_df2, test_list_df), axis=1)
```


```python
test_df = df[df['path'].isin(test_list)]
test_df = test_df.reset_index()
```


```python
test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>mass_size</th>
      <th>asa_score</th>
      <th>op_type_1</th>
      <th>op_type_2</th>
      <th>clinical_stage_t</th>
      <th>clinical_stage_m</th>
      <th>clinical_stage_n</th>
      <th>fuhrman_nuclear_grade</th>
      <th>diameter</th>
      <th>renal_capsule_invasion</th>
      <th>e_gfr</th>
      <th>survival_period</th>
      <th>alive_or_death</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>56.0</td>
      <td>1.0</td>
      <td>26.4</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-B/TA2251-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>51.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>3.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.6</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>68.0</td>
      <td>1.0</td>
      <td>23.4</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-B/TA2251-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>58.0</td>
      <td>1.0</td>
      <td>23.5</td>
      <td>2.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>24.2</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>4.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>434</th>
      <td>2157</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>22.7</td>
      <td>7.4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>6.5</td>
      <td>1.0</td>
      <td>84.4</td>
      <td>69.3</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3289-C/TA3289-...</td>
    </tr>
    <tr>
      <th>435</th>
      <td>2174</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>20.8</td>
      <td>2.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>36.4</td>
      <td>38.2</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3292-B/TA3292-...</td>
    </tr>
    <tr>
      <th>436</th>
      <td>2175</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>20.8</td>
      <td>2.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>36.4</td>
      <td>38.2</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3292-C/TA3292-...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>2185</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>1.8</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>73.1</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3321-A/TA3321-...</td>
    </tr>
    <tr>
      <th>438</th>
      <td>2193</td>
      <td>81.0</td>
      <td>1.0</td>
      <td>21.4</td>
      <td>4.2</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>4.3</td>
      <td>1.0</td>
      <td>58.1</td>
      <td>114.5</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3322-C/TA3322-...</td>
    </tr>
  </tbody>
</table>
<p>439 rows × 18 columns</p>
</div>




```python
test_df['path'][0]
```




    '/workspace/data/parsed_level0/TA2251-B/TA2251-B_01.png'




```python
result_df2['path'][0]
```




    '/workspace/data/parsed_level0/TA3322-C/TA3322-C_56.png'




```python
# final_df1 = pd.merge(test_df,result_df, how='inner', on='path')
# final_df1 = pd.concat((test_df, result_df), axis=1)
final_df2 = pd.merge(test_df,result_df2, how='inner', on='path')
```


```python
final_df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>mass_size</th>
      <th>asa_score</th>
      <th>op_type_1</th>
      <th>op_type_2</th>
      <th>clinical_stage_t</th>
      <th>clinical_stage_m</th>
      <th>...</th>
      <th>fuhrman_nuclear_grade</th>
      <th>diameter</th>
      <th>renal_capsule_invasion</th>
      <th>e_gfr</th>
      <th>survival_period</th>
      <th>alive_or_death</th>
      <th>path</th>
      <th>test_label_list</th>
      <th>test_prob_list</th>
      <th>test_pred_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>56.0</td>
      <td>1.0</td>
      <td>26.4</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-B/TA2251-...</td>
      <td>0</td>
      <td>0.368454</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>51.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>3.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>2.6</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>67.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
      <td>0</td>
      <td>0.416994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>68.0</td>
      <td>1.0</td>
      <td>23.4</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>1.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-B/TA2251-...</td>
      <td>0</td>
      <td>0.404030</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>58.0</td>
      <td>1.0</td>
      <td>23.5</td>
      <td>2.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
      <td>0</td>
      <td>0.260402</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>78.0</td>
      <td>1.0</td>
      <td>24.2</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>4.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>120.0</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA2251-A/TA2251-...</td>
      <td>0</td>
      <td>0.375083</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>434</th>
      <td>2157</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>22.7</td>
      <td>7.4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>6.5</td>
      <td>1.0</td>
      <td>84.4</td>
      <td>69.3</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3289-C/TA3289-...</td>
      <td>1</td>
      <td>0.872973</td>
      <td>1</td>
    </tr>
    <tr>
      <th>435</th>
      <td>2174</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>20.8</td>
      <td>2.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>36.4</td>
      <td>38.2</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3292-B/TA3292-...</td>
      <td>1</td>
      <td>0.587538</td>
      <td>1</td>
    </tr>
    <tr>
      <th>436</th>
      <td>2175</td>
      <td>70.0</td>
      <td>1.0</td>
      <td>20.8</td>
      <td>2.1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>2.5</td>
      <td>0.0</td>
      <td>36.4</td>
      <td>38.2</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3292-C/TA3292-...</td>
      <td>1</td>
      <td>0.536236</td>
      <td>1</td>
    </tr>
    <tr>
      <th>437</th>
      <td>2185</td>
      <td>57.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>1.8</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>1.2</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>73.1</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3321-A/TA3321-...</td>
      <td>1</td>
      <td>0.178476</td>
      <td>0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>2193</td>
      <td>81.0</td>
      <td>1.0</td>
      <td>21.4</td>
      <td>4.2</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1</td>
      <td>4.3</td>
      <td>1.0</td>
      <td>58.1</td>
      <td>114.5</td>
      <td>0.0</td>
      <td>/workspace/data/parsed_level0/TA3322-C/TA3322-...</td>
      <td>1</td>
      <td>0.339014</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>439 rows × 21 columns</p>
</div>




```python
# ImageNet
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ax = plt.subplot(111)

kmf = KaplanMeierFitter()

m_low = final_df1[(final_df1['test_pred_list']  == 0)]
# m_high = final_df1[(final_df1['fuhrman_nuclear_grade'] == 1)]

m_high = final_df1[(final_df1['test_pred_list'] == 1)]

T_l = m_low.survival_period
E_l = m_low.alive_or_death
T_h = m_high.survival_period
E_h = m_high.alive_or_death

# pred label 0
kmf.fit(T_l, event_observed=E_l, label='high')
kmf.plot_survival_function(ax=ax)

# pred label 1
kmf.fit(T_h, event_observed=E_h, label='low')
kmf.plot_survival_function(ax=ax)

plt.title('Survival period through fuhrman grade', fontsize=15)


pvalue = logrank_test(T_l, T_h, E_l, E_h).p_value
plt.text(80,0.8, 'p-value : %s' % float('%.2g' %pvalue))
plt.ylabel('Survival probability')
plt.xlabel('timeline (month)')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-17-a467a361b29a> in <module>
          7 kmf = KaplanMeierFitter()
          8 
    ----> 9 m_low = final_df1[(final_df1['test_pred_list']  == 0)]
         10 # m_high = final_df1[(final_df1['fuhrman_nuclear_grade'] == 1)]
         11 


    NameError: name 'final_df1' is not defined



    
![png](output_21_1.png)
    



```python
# Pancancer
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

ax = plt.subplot(111)

kmf = KaplanMeierFitter()

m_low = final_df2[(final_df2['test_pred_list']  == 0)]
# m_high = test_ddf[(test_ddf['pred']  == 1)]
m_high = final_df2[(final_df2['test_pred_list'] == 1 )]
# m_low = final_df2[(final_df2['fuhrman_nuclear_grade'] == 0)]

# m_high = final_df2[(final_df2['fuhrman_nuclear_grade'] == 1)]

T_l = m_low.survival_period
E_l = m_low.alive_or_death
T_h = m_high.survival_period
E_h = m_high.alive_or_death

# pred label 0
kmf.fit(T_l, event_observed=E_l, label='high')
kmf.plot_survival_function(ax=ax)

# pred label 1
kmf.fit(T_h, event_observed=E_h, label='low')
kmf.plot_survival_function(ax=ax)

plt.title('Survival period through fuhrman grade', fontsize=15)


pvalue = logrank_test(T_l, T_h, E_l, E_h).p_value
plt.text(80,0.8, 'p-value : %s' % float('%.2g' %pvalue))
plt.ylabel('Survival probability')
plt.xlabel('timeline (month)')
```




    Text(0.5, 0, 'timeline (month)')




    
![png](output_22_1.png)
    



```python

```


```python

```
