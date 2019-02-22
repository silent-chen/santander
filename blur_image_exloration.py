import numpy as np
import cv2
import matplotlib.pyplot as plt
list1='./data/split/train_3200_0'
list2='./data/split/train_3200_1'


f=open(list1,'r')
lines1=f.readlines()
f.close()
f=open(list2,'r')
lines2=f.readlines()
f.close()
lines=[]
for l in lines2:
    if l not in lines1:
        lines.append(l)

print(len(lines))
f=open('./data/split/train_3200_1but0','w')
for l in lines:
    f.write(l)
f.close()
exit(0)
image=[]
mask=[]
for l in lines2:
    image.append('./data/train/images/'+l.split('/')[-1].strip()+'.png')
    mask.append('./data/train/masks/'+l.split('/')[-1].strip()+'.png')
print(image)
print(mask)
imgs=[]
masks=[]
for img_,mask_ in zip(image,mask):
    img=cv2.imread(img_)
    mask=cv2.imread(mask_)
    imgs.append(img)
    masks.append(mask)
masks=np.array(masks)
# masks[masks==255]= 0
print(np.max(masks))

a=np.reshape(masks,(1,-1))
a=np.squeeze(a)
plt.hist(a, bins=255)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()
# print(imgs[0])
# print(masks[0])
# blur=[]
# for img in imgs:
#     blur.append(cv2.Laplacian(img, cv2.CV_64F).var())
#import matplotlib.pyplot as plt
#plt.hist(blur, normed=True, bins=5)
#plt.ylabel('Probability')
#plt.show()
# blur=np.array(blur).reshape(-1,1)
# print(blur)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=10, random_state=0).fit(blur)
# print(kmeans.labels_)