import gzip
import numpy as np
import struct
import torch

def load_data(filename):
	with open(filename, 'rb') as file:
		data = file.read()
		fmt_header = '>iiii'
		offset=0
		magic, n, rows, cols= struct.unpack_from(fmt_header, data, offset)
		print('magic:%d, count: %d, size: %d*%d' % (magic, n, rows, cols))
		size = rows * cols
		offset += struct.calcsize(fmt_header)  
		fmt_image = '>' + str(size) + 'B'  
		images = np.empty((n, size))
		for i in range(n):
			images[i] = np.array(struct.unpack_from(fmt_image, data, offset))/255
			offset += struct.calcsize(fmt_image)
	return images

def load_label(filename):
	with open(filename, 'rb') as f:
		data = f.read()
		fmt_header='>ii'
		offset= 0
		magic, n = struct.unpack_from(fmt_header, data, offset)
		print('magic: %d, count: %d'%(magic, n))
		offset += struct.calcsize(fmt_header)
		labels = np.zeros(n)
		fmt_label = '>B'
		for i in range(n):
			#j = int(struct.unpack_from(fmt_label, data, offset)[0])
			#labels[i][j] = 1
			labels[i] = struct.unpack_from(fmt_label, data, offset)[0]
			offset += struct.calcsize(fmt_label)
	return labels

train_images = load_data('train-images.idx3-ubyte')
train_labels = load_label('train-labels.idx1-ubyte')
test_images = load_data('t10k-images.idx3-ubyte')
test_labels = load_label('t10k-labels.idx1-ubyte')



#print(train_images[0])
#print(train_labels[0])

class TwoLayerNet(torch.nn.Module):  
	def __init__(self, D_in, H1, H2, D_out):
		super(TwoLayerNet, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H1, bias=False)
		#self.linear2 = torch.nn.Linear(H1, H2, bias=False)
		self.linear2 = torch.nn.Linear(H1, D_out, bias=False)

		#self.linear3 = torch.nn.Linear(H2, D_out, bias=False)
		self.softmax = torch.nn.LogSoftmax(dim=1)
		#self.linear3
	
	def forward(self, x):
		#y_pred = self.linear3(self.linear2(self.linear1(x).clamp(min=0)).clamp(min=0))
		y_pred = self.softmax(self.linear2(self.linear1(x).clamp(min=0)).clamp(min=0).clamp(min=0))
		
		return y_pred

batch_size=100
D_in = 28*28
H1 = 300
H2 = 200
D_out = 10
n = 60000

model = TwoLayerNet(D_in, H1, H2, D_out)
#cross_loss = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.NLLLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for it in range(500):
	loss_ite = 0
	for i in range(0,10000,batch_size):
		y = torch.from_numpy(train_labels[i:i+batch_size])
		x = torch.from_numpy(train_images[i:i+batch_size])
		#y = torch.from_numpy(train_labels[:200])
		#x = torch.from_numpy(train_images[:200])
		y = torch.tensor(y, dtype=torch.long)
		x = torch.tensor(x, dtype=torch.float32)
		y_pred = model(x)
		#pred = out.max(1)
		#print(y_pred, y)
		loss = loss_fn(y_pred, y)
		loss_ite += loss.item()
		#print(loss.item())
		optimizer.zero_grad()
		loss.backward()
		with torch.no_grad():
			optimizer.step()
	print(it, loss_ite/batch_size)


y_test = torch.from_numpy(test_labels[:100])
x_test = torch.from_numpy(test_images[:100])
y_test = torch.tensor(y_test, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test_pred = model(x_test)
loss = loss_fn(y_test_pred, y_test)
print('test loss:', loss.item())