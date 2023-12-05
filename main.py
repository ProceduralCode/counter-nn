
import sys
import csv
from pathlib import Path
import numpy as np
import cv2
import cnn.testing as cnn
import dm.density_map as dm

class Dataset:
	data_dir = Path('data') / 'bee_dataset'

	def __init__(self, test_size=0.15):
		metadata = {}
		for source_dir in Dataset.data_dir.iterdir():
			# print(source_dir)
			print(f"Loading {source_dir.name}...", end='\r')
			csv_file = source_dir / (source_dir.name + '.csv')
			with open(csv_file) as f:
				reader = csv.reader(f)
				next(reader) # skip header
				for row in reader:
					img = source_dir.name + '/' + row[0]
					if img not in metadata:
						metadata[img] = []
					point_data = eval(row[5])
					metadata[img].append((point_data['cx'], point_data['cy']))
		print(' '*50, end='\r')
		metadata = [(k, v) for k, v in metadata.items()]
		np.random.shuffle(metadata)
		split_idx = int(len(metadata) * test_size)
		self.metadata_test = metadata[:split_idx]
		self.metadata_train = metadata[split_idx:]

	def iter(self, train=True, batch_size=10):
		metadata = self.metadata_train if train else self.metadata_test
		np.random.shuffle(metadata)
		# for img, points in metadata:
		# 	file_path = Dataset.data_dir / img
		# 	img = cv2.imread(str(file_path)).astype(np.float32) / 255
		# 	yield img, points
		for i in range(0, len(metadata), batch_size):
			batch = metadata[i:i+batch_size]
			imgs = []
			points = []
			for img, point in batch:
				file_path = Dataset.data_dir / img
				img = cv2.imread(str(file_path)).astype(np.float32) / 255
				imgs.append(img)
				points.append(point)
			yield np.stack(imgs), points

def show_img(img, points=None, heatmap=None):
	img = img.copy()
	if points is not None:
		for x, y in points:
			cv2.line(img, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
			cv2.line(img, (x+5, y-5), (x-5, y+5), (0, 0, 255), 1)
	if heatmap is not None:
		heatmap = heatmap / np.max(heatmap)
		heatmap = np.clip(heatmap, 0, 1)
		heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET).astype(np.float32) / 255
		img = cv2.addWeighted(img, 1, heatmap, 0.5, 0)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	if len(sys.argv) < 2 or sys.argv[1] not in ['cnn', 'dm']:
		print('Usage: python main.py [cnn|dm]')
	elif sys.argv[1] == 'cnn':
		cnn.main()
	elif sys.argv[1] == 'dm':
		dm.main()

if __name__ == '__main__':
	main()
