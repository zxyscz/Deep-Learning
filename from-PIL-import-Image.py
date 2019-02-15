from PIL import Image
import glob,os

# cnt = 10
# for filename in glob.glob('/Users/ling/Desktop/DUO/*.JPG'):
# 	print filename
# 	new_name = str(cnt)+'.JPG'
# 	os.rename(filename, new_name)
# 	cnt+=1

mw = 100
ms = 20

msize = mw*ms

toImage = Image.new('RGBA',(2000,2000))

for y in range(1,21):
    for x in range(1,21):
	try:
		i=x+y
		fromImage = Image.open(r"/Users/ling/Desktop/DUO/%s.jpg" %str(i))
		fromImage = fromImage.resize((100,100),Image.ANTIALIAS)
		toImage.paste(fromImage,((x-1)*mw,(y-1)*mw))
	except IOError:
		pass

toImage.show()
toImage.save('/Users/ling/Desktop/DUO/LX.png')