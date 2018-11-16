


import glob
import os


files = glob.glob('image*.png')

image_num=0
for index, file in enumerate(files):
	new_file0 = file.split('_')[0]+'_'+str(image_num).zfill(4)+'.png'
	new_file1 = file.split('_')[0]+'_'+str(image_num+1).zfill(4)+'.png'
	os.system("cp "+file+" ./trial3/"+new_file0)
	os.system("cp "+file+" ./trial3/"+new_file1)
	image_num+=2
