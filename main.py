
import csv
from pathlib import Path
import numpy as np
import cv2

def main():
	dataset = Dataset(test_size=0.15)
	for img, count in dataset.iter(train=True):
		print(img.shape, count)
		show_img(img)
		break

class Dataset:
	data_dir = Path('data')
	imgs_dir = data_dir / 'coins_images' / 'coins_images'

	def __init__(self, test_size=0.15):
		with open(self.data_dir / 'coins_count_values.csv') as f:
			reader = csv.reader(f)
			next(reader) # skip header (folder,image_name,coins_count)
			metadata = list(reader)
		np.random.shuffle(metadata)
		split_idx = int(len(metadata) * test_size)
		self.metadata_test = metadata[:split_idx]
		self.metadata_train = metadata[split_idx:]

	def iter(self, train=True, preprocess=True):
		metadata = self.metadata_train if train else self.metadata_test
		np.random.shuffle(metadata)
		for folder, image_name, coins_count in metadata:
			file_path = self.imgs_dir / folder / image_name
			img = cv2.imread(str(file_path)).astype(np.float32) / 255
			if preprocess:
				img = self.preprocess(img)
			yield img, coins_count

	def preprocess(self, img):
		flip = np.random.randint(0, 2)
		rotate = np.random.randint(0, 4)
		if flip:
			img = np.flip(img, axis=1)
		if rotate:
			img = np.rot90(img, rotate)
		return img

def show_img(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
