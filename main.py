
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

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

class Model(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.pool1 = nn.MaxPool2d(2)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.pool2 = nn.MaxPool2d(2)
		self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
		# self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		# self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
		# self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
		# self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
		# self.conv9 = nn.Conv2d(64, 32, 3, padding=1)
		# self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
		# self.conv11 = nn.Conv2d(32, 1, 1)
		self.conv7 = nn.Conv2d(128, 1, 1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x1 = F.relu(self.conv2(F.relu(self.conv1(x))))
		x2 = F.relu(self.conv4(F.relu(self.conv3(self.pool1(x1)))))
		x = F.relu(self.conv6(F.relu(self.conv5(self.pool2(x2)))))
		# x = F.relu(self.conv8(F.relu(self.conv7(torch.cat([x2, self.up1(x)], dim=1)))))
		# x = F.relu(self.conv10(F.relu(self.conv9(torch.cat([x1, self.up2(x)], dim=1)))))
		# x = self.conv11(x)
		x = self.conv7(x)
		x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
		return x

def em():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = Model().to(device)
	dataset = Dataset(test_size=0.15)
	for epoch in range(100):
		model.eval()
		model.zero_grad()
		losses = []
		for i, (imgs, pointss) in enumerate(dataset.iter(train=True)):

			# Construct heatmap
			gauss_heatmaps = []
			for img, points in zip(imgs, pointss):
				gauss_heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
				dot_size = 10
				for x, y in points:
					gauss_heatmap[y-1][x-1] = 1
				gauss_heatmap = cv2.GaussianBlur(gauss_heatmap, (dot_size*6-1, dot_size*6-1), dot_size)
				gauss_heatmaps.append(gauss_heatmap)
			gauss_heatmaps = np.stack(gauss_heatmaps)

			# Forward (predict heatmap)
			input = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
			output = model(input)

			# Loss
			gt = torch.from_numpy(gauss_heatmaps).unsqueeze(0).to(device)
			loss = torch.mean((output - gt)**2)
			losses.append(loss.item())

			# Backprop (diff from gt heatmap)
			learning_rate = 0.01
			loss.backward()
			with torch.no_grad():
				for param in model.parameters():
					param -= learning_rate * param.grad

			if i == 10:
				break

		show_img(imgs[0], pointss[0], heatmap=output[0].detach().cpu().permute(1, 2, 0).numpy())
		show_img(imgs[0], pointss[0], heatmap=gauss_heatmaps[0])
		show_img(imgs[0], pointss[0])

		print('Epoch: {}, Loss: {}'.format(epoch, np.mean(losses)))
		show_img(imgs[0], heatmap=output[0].detach().cpu().permute(1, 2, 0).numpy())

def main():
	em()

if __name__ == '__main__':
	main()
