import glob
import shutil
i=0
for name in glob.glob('D:/covi_cvpr/val/non-covid/*/*.jpg'):
	# print(name)
	shutil.move(name, "D:/covi_cvpr/val_1/non_covid/{}.jpg".format(i))
	i= i+1
	if i%500==0 :
		print(i)