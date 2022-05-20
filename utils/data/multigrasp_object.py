import os
import glob
origin_path = os.getcwd()#记录一下原始的路径
os.chdir('/home/sam/Desktop/multiobject/rgbd')#这是我的路径
print(os.listdir())
path='/home/sam/Desktop/multiobject/rgbd'
depth = glob.glob(os.path.join(path,'depth_*.png'))
depth.sort()
print(depth[0:10])

rgb=glob.glob(os.path.join(path,'rgb*.jpg'))
rgb.sort()
print(rgb[0:10])

label=glob.glob(os.path.join(path,'rgb_*.txt'))
label.sort()
print(label[0:10])

from PIL import Image
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,15))
# for i in range(9):
#     img = Image.open(rgb[i])
#     plt.subplot(331+i)
#     plt.imshow(img)
# plt.show()
# plt.figure(figsize=(15,15))
# for i in range(9):
#     img = Image.open(depth[i])
#     plt.subplot(331+i)
#     plt.imshow(img)
# plt.show()

# with open(label[0],'r') as f:
#     grasp_data = f.read()
# print(grasp_data)
# grasp_data = [grasp.strip() for grasp in grasp_data]#去除末尾换行符

def str2num(point):
    '''
    :参数  :point,字符串，以字符串形式存储的一个点的坐标
    :返回值 :列表，包含int型抓取点数据的列表[x,y]

    '''
    x, y = point.split()
    x, y = int(round(float(x))), int(round(float(y)))

    return (x, y)


def get_rectangle(cornell_grasp_file):
    '''
    :参数  :cornell_grap_file:字符串，指向某个抓取文件的路径
    :返回值 :列表，包含各个抓取矩形数据的列表

    '''
    grasp_rectangles = []
    with open(cornell_grasp_file, 'r') as f:
        while True:
            grasp_rectangle = []
            point0 = f.readline().strip()
            if not point0:
                break
            point1, point2, point3 = f.readline().strip(), f.readline().strip(), f.readline().strip()
            grasp_rectangle = [str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)]
            grasp_rectangles.append(grasp_rectangle)

    return grasp_rectangles
i=90
grs = get_rectangle(label[i])
print(grs)
import cv2
import random
import cv2
img = cv2.imread(rgb[i])
for gr in grs:
    # 产生随机颜色
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # 绘制添加矩形框
    for i in range(3):  # 因为一个框只有四条线，所以这里是3
        img = cv2.line(img, gr[i], gr[i + 1], color, 3)
    img = cv2.line(img, gr[3], gr[0], color, 2)  # 补上最后一条封闭的线

plt.figure(figsize=(10, 10))
plt.imshow(img)  # 之前用cv2.imshow，显示倒是能显示，就是服务老是挂掉，现在索性换成这个
plt.show()
plt.imshow(img)
plt.show()