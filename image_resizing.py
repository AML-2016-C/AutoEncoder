from PIL import Image
import glob,os

if(__name__=="__main__"):
	count=1
	files = glob.glob("Dataset/*");
	for folder in files:
		file_list = glob.glob(folder+"/*")
		foldername = folder.split("\\")[1];
		for image in file_list:
			print count;
			filename = image.split("\\")[2];
			foo = Image.open(image)
			foo = foo.resize((16,16),Image.ANTIALIAS)
			if not os.path.exists("Normalized\\"+foldername):
				os.makedirs("Normalized\\"+foldername)
			foo.save("Normalized\\"+foldername+"\\"+filename,optimize=True,quality=95)
			count+=1;
