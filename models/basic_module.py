import torch as t
import time

class BasicModule(t.nn.Module):

	def __init__(self):
		super(self,BasicModule).__init__()
		self.model_name = str(type(self))

	def load(self,path):
		self.load_state_dict(t.load(path))

	def save(self, name=None):
		#model name + time
		if name is None: 
			prefix = 'checkpoints/' + self.model_name + '_'
			name = time.strftime(prefix + '%m%d_%H: %M:%S.pth')
		t.save(self.state_dict(), name)
		return name
	def get_optimizer(self, lr, weight_decay):
		return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

