import torch.nn as nn
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
# 打印网络的结构
# print(model)
# params=model.state_dict()
# for k,v in params.items():
#     print(k)    #打印网络中的变量名
# # print(params['conv1.weight'])   #打印conv1的weight
# print(params['conv1.bias'])   #打印conv1的bias
# params['conv1.bias']+=1
# print(params['conv1.bias'])
# params=model.state_dict()
# print(params['conv1.bias'])

params=model.parameters()
for p in params:
    print(p.data)
    p.data+=1
params=model.parameters()
for p in params:
    print(p.data)
f=open('./data/split/train_3600_1_origin','r')
lines=f.readlines()
fo=open('./data/split/train_800_1_origin','w')
for l in lines[:800]:
    fo.write(l.strip()+'\n')
fo.close()
f.close()
