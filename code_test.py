from resnet34.model_resnet34 import  SaltNet as Net
import torch
net=Net().cuda()
# print(net)
# print(net.encoder)
net.load_state_dict(torch.load('E:\\DHWorkStation\\Project\\tgs_pytorch\\output\\resnet34\\checkpoint\\00008000_model.pth'))