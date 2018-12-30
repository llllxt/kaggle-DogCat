import warnings
import torch as t

class Default(object):
	env = 'default'
	vis_port = 8097
	model = 'resnet34'

	train_data_root = './data/train/'
	test_data_root = './data/test/'
	load_model_path = None

	batch_size = 32
	use_gpu = True
	num_workers = 4
	print_freq = 20

	max_epoch = 10
	lr = 0.001
	lr_decay = 0.5
	weight_decay = 0e-5

	def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig() 