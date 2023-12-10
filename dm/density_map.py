
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from main import Dataset

def show_img(img, points=None, heatmap=None, heatmap_perc=0.5):
	img = img.copy()
	if points is not None:
		for x, y in points:
			cv2.line(img, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
			cv2.line(img, (x+5, y-5), (x-5, y+5), (0, 0, 255), 1)
	if heatmap is not None:
		heatmap = heatmap / np.max(heatmap)
		heatmap = np.clip(heatmap, 0, 1)
		heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET).astype(np.float32) / 255
		img = cv2.addWeighted(img, 1, heatmap, heatmap_perc, 0)
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

class Model(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
		self.pool1 = nn.MaxPool2d(2)
		self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
		self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
		self.pool2 = nn.MaxPool2d(2)
		self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
		self.conv9 = nn.Conv2d(128, 128, 3, padding=1)
		self.conv10 = nn.Conv2d(128, 1, 1)

		# self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		# self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		# self.pool1 = nn.MaxPool2d(2)
		# self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		# self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		# self.pool2 = nn.MaxPool2d(2)
		# self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
		# self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
		# self.conv7 = nn.Conv2d(128, 1, 1)

		# self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
		# self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
		# self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
		# self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
		# self.conv9 = nn.Conv2d(64, 32, 3, padding=1)
		# self.conv10 = nn.Conv2d(32, 32, 3, padding=1)
		# self.conv11 = nn.Conv2d(32, 1, 1)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x1 = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))))
		x2 = F.relu(self.conv6(F.relu(self.conv5(F.relu(self.conv4(self.pool1(x1)))))))
		x = F.relu(self.conv9(F.relu(self.conv8(F.relu(self.conv7(self.pool2(x2)))))))
		x = self.conv10(x)

		# x1 = F.relu(self.conv2(F.relu(self.conv1(x))))
		# x2 = F.relu(self.conv4(F.relu(self.conv3(self.pool1(x1)))))
		# x = F.relu(self.conv6(F.relu(self.conv5(self.pool2(x2)))))
		# x = self.conv7(x)

		# x = F.relu(self.conv8(F.relu(self.conv7(torch.cat([x2, self.up1(x)], dim=1)))))
		# x = F.relu(self.conv10(F.relu(self.conv9(torch.cat([x1, self.up2(x)], dim=1)))))
		# x = self.conv11(x)

		x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
		return x

	def save(self, path, metadata):
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		state_dict = self.state_dict()
		state_dict['metadata'] = metadata
		torch.save(state_dict, path)

	def load(self, path):
		state_dict = torch.load(path)
		metadata = state_dict.pop('metadata')
		self.load_state_dict(state_dict)
		return metadata

def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = Model().to(device)
	dataset = Dataset(test_size=0.15)
	learning_rate = 0.01

	# Load model
	save_path = 'dm/saves/model.pth'
	metadata = { 'epochs': [], 'losses': [], 'errors': [], }
	if os.path.exists(save_path):
		metadata = model.load(save_path)

	try:
		while True:
			# Train
			model.train()
			model.zero_grad()
			losses = []
			for i, (imgs, pointss) in enumerate(dataset.iter(train=True)):
				print(f"Training batch {i}", end='\r')

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
				loss.backward()
				with torch.no_grad():
					for param in model.parameters():
						param -= learning_rate * param.grad

				# if i == 50:
				# 	break
			print(" "*50, end='\r')
			loss = np.mean(losses)

			# Test
			model.eval()
			errors = []
			for i, (imgs, pointss) in enumerate(dataset.iter(train=False)):
				print(f"Testing batch {i}", end='\r')
				input = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
				output = model(input)
				output = output.detach().cpu().permute(0, 2, 3, 1).numpy()
				counts = np.sum(output, axis=(1, 2, 3))
				for points, count in zip(pointss, counts):
					errors.append(np.abs(len(points) - count))
			print(" "*50, end='\r')
			error = np.mean(errors)

			# Save
			epoch = len(metadata['epochs'])
			print(f'Epoch: {epoch}, Loss: {loss*10e6:.2f}, Error: {error:.2f}')
			metadata['epochs'].append(epoch)
			metadata['losses'].append(loss)
			metadata['errors'].append(error)
			model.save(save_path, metadata)

	except KeyboardInterrupt:
		imgs, pointss = next(dataset.iter(train=False))
		input = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
		output = model(input)
		output = output.detach().cpu().permute(0, 2, 3, 1).numpy()
		show_img(imgs[0], pointss[0])
		show_img(imgs[0], pointss[0], heatmap=output[0])
		show_img(imgs[0], pointss[0], heatmap=output[0], heatmap_perc=1)
