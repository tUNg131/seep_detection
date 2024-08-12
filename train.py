import os

import cv2
import torch
import torch.nn as nn
import torch.utils.data as data

from data import SeepDataset
from model import UNet

num_classes = 8
epochs = 100
train_batch_size = 32
val_batch_size = 64
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seep_dataset = SeepDataset("data/train_images_256/", "data/train_masks_256/")
train_set, val_set, test_set = data.random_split(
		seep_dataset, [0.7, 0.2, 0.1]
)

train_loader = data.DataLoader(train_set,
															 batch_size=train_batch_size,
															 shuffle=True)

val_loader = data.DataLoader(val_set,
														 batch_size=val_batch_size,
														 shuffle=False)

model = UNet(1, num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_one_epoch(epoch_index):
		running_loss = 0.
		cnt = 0

		for i, data in enumerate(train_loader):
				inputs, labels, _ = data
				
				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()

				outputs = model(inputs)

				loss = loss_fn(outputs, labels)
				loss.backward()

				optimizer.step()

				running_loss += loss.item()
				cnt += 1
		train_loss = running_loss / cnt
		return train_loss


best_val_loss = float('inf')

for epoch_index in range(1, epochs + 1):
		model.train(True)
		train_loss = train_one_epoch(epoch_index)

		val_loss = 0.
		cnt = 0

		model.eval()
		with torch.no_grad():
				for i, vdata in enumerate(val_loader):
						vinputs, vlabels, _ = vdata

						vinputs = vinputs.to(device)
						vlabels = vlabels.to(device)

						voutputs = model(vinputs)
						vloss = loss_fn(voutputs, vlabels)

						val_loss += vloss.item()
						cnt += 1
		
		val_loss /= cnt	

		if val_loss < best_val_loss:
				try:
						os.remove(f"model/unet_{best_val_loss}")
				except:
						pass
				best_val_loss = val_loss
				os.makedirs("model", exist_ok=True)
				path = f"model/unet_{best_val_loss}"
				torch.save(model.state_dict(), path)
		print(f"Epoch {epoch_index} | train_loss: {train_loss} | val_loss: {val_loss}")

# Testing
model.load_state_dict(torch.load(f"model/unet_{best_val_loss}"))
print(f"Loaded unet_{best_val_loss}")

test_loss = 0.
cnt = 0

model.eval()

os.makedirs("output", exist_ok=True)

with torch.no_grad():
		for i, tdata in enumerate(val_loader):
				tinputs, tlabels, img_ids = tdata

				tinputs = tinputs.to(device)
				tlabels = tlabels.to(device)

				toutputs = model(tinputs)
				
				predictions = torch.argmax(toutputs, dim=1)
				predictions = torch.chunk(predictions, chunks=val_batch_size, dim=0)
				
				for pred, img_id in zip(predictions, img_ids):
						pred = torch.squeeze(pred, dim=0)
						cv2.imwrite(f"output/{img_id}", pred.cpu().numpy())

				tloss = loss_fn(toutputs, tlabels)
				test_loss += tloss.item()
				cnt += 1		
		test_loss /= cnt
print(f"Training completed | test_loss: {test_loss}")
