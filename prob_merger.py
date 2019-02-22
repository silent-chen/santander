import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '3,2' #'3,2,1,0'

from data_util import *
out_dir = \
    'E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\seresnext50_bn\\oc_test\\'

# split, mode = 'valid_400_1_origin', 'valid'
split, mode  = 'test_18000',  'test'
csv_file = out_dir + '/test/merged_submit-%s-aug2.csv' % (split)
prob_merger=[

]
threshold=0.5
total_sum=len(prob_merger)
for index,pred in enumerate(prob_merger):
    if index==0:
        all_prob=np.load(pred).astype(np.float32) / 255
    else:
        all_prob+=np.load(pred).astype(np.float32) / 255
all_prob/=total_sum

all_prob = all_prob > threshold
print(all_prob.shape)

# ----------------------------

split_file = 'E:\\DHWorkStation\\Project\\tgs_pytorch\\data/split/' + split
lines = read_list_from_file(split_file)

id = []
rle_mask = []
for n, line in enumerate(lines):
    folder, name = line.split('/')
    id.append(name)

    if (all_prob[n].sum() <= 0):
        encoding = ''
    else:
        encoding = run_length_encode(all_prob[n])
    assert (encoding != [])

    rle_mask.append(encoding)

df = pd.DataFrame({'id': id, 'rle_mask': rle_mask}).astype(str)
df.to_csv(csv_file, index=False, columns=['id', 'rle_mask'], encoding='utf-8')
print('submit done')
