import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage import filters

a1 = np.arange(0,12).reshape(3,4)
a2 = np.mean(a1, axis=1)
print("每行均值为",a2)

b1 = np.eye(5)
n1 = np.array(b1[1,:])
n2 = np.array(b1[2,:])
b1[1,:] = n2
b1[2,:] = n1
print("\n单位阵交换后如下")
print(b1)

c1 = np.random.randint(10,51, (4,5))
print("\n生成随机整数矩阵如下")
print(c1)
for i in range(4):
    for j in range(5):
        if c1[i,j]>40:
            print("大于40的位置为[{},{}], 值为{}".format(i+1,j+1,c1[i,j]))



plt.figure(num='2.1', figsize=(12,12))
img = data.cat()
img1 = rgb2gray(img)
plt.subplot(1,2,1), plt.title('Cat'), plt.imshow(img)
plt.subplot(1,2,2), plt.title('Cat gray'), plt.imshow(img1, cmap=plt.cm.gray)
plt.show()


plt.figure(num='2.2', figsize=(12,12))
img2 = filters.sobel(img1)
plt.subplot(1,2,1), plt.title('Cat'), plt.imshow(img)
plt.subplot(1,2,2), plt.title('Cat Sobel'), plt.imshow(img2, cmap=plt.cm.gray)
plt.show()


plt.figure(num='2.3.1', figsize=(12,12))
NUM = 10
length = int(img.shape[0]/NUM)
width = int(img.shape[1]/NUM)

now_len = 0
now_wid = 0
x = 1
img4 = []
for i in range(NUM):
    for j in range(NUM):
        plt.subplot(NUM, NUM, x), plt.imshow(img[now_len:now_len+length, now_wid:now_wid+width, :]),plt.axis('off')
        img4.append(img[now_len:now_len+length, now_wid:now_wid+width, :])
        now_wid = now_wid+width
        x+=1
    now_len=now_len+length
    now_wid = 0

plt.show()


random.shuffle(img4)
img_height, img_width, channels = img4[0].shape
rows, cols = NUM, NUM
merged_img = np.zeros((rows * img_height, cols * img_width, channels), dtype=np.uint8)

# 将每张图片放置到大图的相应位置
for i in range(rows):
    for j in range(cols):
        start_row = i * img_height
        start_col = j * img_width
        merged_img[start_row:start_row + img_height, start_col:start_col + img_width, :] = img4[i * cols + j]

plt.figure(figsize=(12, 12))
plt.imshow(merged_img)
plt.axis('off')
plt.show()