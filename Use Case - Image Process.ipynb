# coding=UTF-8<code>
from PIL import Image
import glob,os
import numpy as np
import time

# # Change name of the JPG files
# cnt = 10
# for filename in glob.glob('/Users/ling/Desktop/DUO/*.JPG'):
# 	print filename
# 	new_name = str(cnt)+'.JPG'
# 	os.rename(filename, new_name)
# 	cnt+=1

# # Resize and paste together
# mw = 100
# ms = 20
# msize = mw*ms
# toImage = Image.new('RGBA',(2000,2000))
# for y in range(1,21):
#     for x in range(1,21):
# 	try:
# 		i=x+y
# 		fromImage = Image.open(r"/Users/ling/Desktop/DUO/%s.jpg" %str(i))
# 		fromImage = fromImage.resize((100,100),Image.ANTIALIAS)
# 		toImage.paste(fromImage,((x-1)*mw,(y-1)*mw))
# 	except IOError:
# 		pass
# toImage.show()
# toImage.save('/Users/ling/Desktop/DUO/LX.png')


# # Gradient
# def image(sta,end,depths=10):
#     a = np.asarray(Image.open(sta).convert('L')).astype('float')
#     depth = depths  # 深度的取值范围(0-100)，标准取10
#     grad = np.gradient(a)  # 取图像灰度的梯度值
#     grad_x, grad_y = grad  # 分别取横纵图像梯度值
#     grad_x = grad_x * depth / 100.#对grad_x值进行归一化
#     grad_y = grad_y * depth / 100.#对grad_y值进行归一化
#     A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
#     uni_x = grad_x / A
#     uni_y = grad_y / A
#     uni_z = 1. / A
#     vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
#     vec_az = np.pi / 4.  # 光源的方位角度，弧度值
#     dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对x 轴的影响
#     dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对y 轴的影响
#     dz = np.sin(vec_el)  # 光源对z 轴的影响
#     b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 光源归一化
#     b = b.clip(0, 255)
#     im = Image.fromarray(b.astype('uint8'))  # 重构图像
#     im.save(end)

# def main():
#     xs=10
#     start_time = time.clock()
#     # startss = os.listdir(r"/Users/ling/Desktop/DUO")
#     startss =[ str(i)+".JPG" for i in range(1,10)]
#     time.sleep(2)

#     for starts in startss:
#         start = ''.join(starts)
#         sta = '/Users/ling/Desktop/DUO/' + start
#         end = '/Users/ling/Desktop/DUO/' + 'HD_' + start
#         image(sta=sta,end=end,depths=xs)

#     end_time = time.clock()
#     print('程序运行了  ----' + str(end_time - start_time) + '   秒')
#     time.sleep(3)

# main()

import cv2
from PIL import Image,ImageFilter
## Filter - 使用方法：改变图片的路径选择对应方法即可；采用lut方法时需temp的参数选择模型展示效果
class pic_imshow:
    def __init__(self,path,temp='',params=12):
        self.path=path
        self.temp=temp
        self.params=params    

    # (3)利用PIL中函数实现
    def lut_PIL(self):
        src=Image.open(self.path)
        im2 = src.filter(ImageFilter.BLUR)  # 模糊滤镜
        im2.save("duo_1.jpg")

        im2 = src.filter(ImageFilter.EMBOSS)  # 浮雕效果滤镜
        im2.save("duo_2.jpg")

        im2 = src.filter(ImageFilter.EDGE_ENHANCE)  # 凸显边界
        im2.save("duo_3.jpg")

        im2 = src.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 加倍凸显边界
        im2.save("duo_4.jpg")

        im2 = src.filter(ImageFilter.FIND_EDGES)  # 只保留边界
        im2.save("duo_5.jpg")

        im2 = src.filter(ImageFilter.CONTOUR)  # 铅笔画效果
        im2.save("duo_6.jpg")

        im2 = src.filter(ImageFilter.SMOOTH_MORE)  # 平滑滤镜(阀值更大)
        im2.save("duo_7.jpg")

    # （2）利用opencv中的函数实现
    def lut_opencv(self):
        src=cv2.imread(self.path)
        cv2.namedWindow('input',cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input',src)
        dst=cv2.applyColorMap(src,self.temp)
        cv2.imshow('output',dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()  

    # （1）实现“流年”滤镜的效果:改变通道值的方法
    def fleeting(self):
        src=Image.open(self.path)
        src.show()
        img=np.asarray(Image.open(self.path).convert('RGB'))
        img1=np.sqrt(img*[1.0,0.0,0.0])*self.params
        img2=img*[0.0,1.0,1.0]
        img=img1+img2
        img=Image.fromarray(np.array(img).astype('uint8'))
        # img.show()
        img.save("duo_9.jpg")

    # （1）实现“旧电影”滤镜的效果:改变通道值的方法
    def oldFilm(self):
        src=Image.open(self.path)
        src.show()
        img=np.asarray(Image.open(self.path).convert('RGB'))
        # r=r*0.393+g*0.769+b*0.189 g=r*0.349+g*0.686+b*0.168 b=r*0.272+g*0.534b*0.131
        trans = np.array([[0.393,0.769,0.189],[0.349,0.686,0.168],[0.272,0.534,0.131]]).transpose()
        # clip 超过255的颜色置为255
        img = np.dot(img,trans).clip(max=255)               
        img=Image.fromarray(np.array(img).astype('uint8')) 
        # img.show()
        img.save("duo_10.jpg")
 
if __name__=='__main__':
    path='/Users/ling/Desktop/DUO/1.jpg'
    test=pic_imshow(path,cv2.COLORMAP_COOL)
    # cv2.COLORMAP_COOL可以替换为 cv2.COLORMAP_AUTUMN等或者数字
    # test.lut_PIL()
    # test.lut_opencv()
    # test.fleeting()
    test.oldFilm()
