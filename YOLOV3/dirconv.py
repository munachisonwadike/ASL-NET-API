#helper code to translate files from one directory into another
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

# print(dir_path)

fp = open('./data/coco_16img.txt', 'r')
fp2 = open('./data/coco_16img2.txt', 'w')

for line in fp:
	# print(line)
	line = line.replace("../coco/images", dir_path+"/data/images")
	fp2.write(line)


fp.close()
fp2.close()


os.system('mv ./data/coco_16img2.txt ./data/coco_16img.txt')
