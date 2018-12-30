import fire

from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader

def train(**kwargs):
	opt._parse(kwargs)

	# step1: configure model
	model = getattr(models, opt.model)()
	if opt.load_model_path:
		model.load(opt.load_model_path)
	if opt.use_gpu: model.cuda()

	# step2: data
	train_data = DogCat(opt.train_data_root, train = True)
	val_data = DogCat(opt.train_data_root, train = False)
	train_dataloader = DataLoader(train_data, opt.batch_size,shuffle = True, num_workers = opt.num_workers)
	val_dataloader = DataLoader(val_data, opt.batch_size,shuffle=False,num_workers=opt.num_workers)

	# step3: loss & optimizer

	critetion = t.nn.CrossEntropyLoss()
	lr = opt.lr
	optimizer = t.optim.Adam(model.parameters(),
		lr = lr,
		weight_decay = opt.weight_decay)

	

    # train
    for epoch in range(opt.max_epoch):
    	loss_meter.reset()
    	confusion_matrix.reset()


    	for step,(data,label) in enumerate(train_dataloader):
    		data = Variable(data)
    		label = Variable(label)
    		if opt.use_gpu:
    			data = data.cuda()
    			label = label.cuda()
    		optimizer.zero_grad()
    		score = model(data)
    		loss = critetion(score, label)
    		loss.backward()
    		optimizer.step()

    	model.save()

def val(model, dataloader):
	model.eval()

	for step, data in enumerate(dataloader):
		input, label = DataLoader
		val_input = Variable(input, volatile = True)
		val_label = Variable(label.long(), volatile=True)
		if opt.use_gpu:
			val_input = val_input。cuda()
			val_label = val_label.cuda()
		score = model(val_input)

	model.train()


def test(**kwargs):
	opt.parse(kwargs)

	#model
	model = getattr(models, opt.model).().eval()
	if opt.load_model_path:
		model.load(opt.load_model_path)
	if opt.use_gpu:
		model.cuda()

	#data
	train_data = DogCat(opt.test_data_root, test=True)
	test_dataloader = DataLoader(train_data,,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)

	results = []

	for step, (data,path) in enumerate(test_dataloader):
	input = t.autograd.Variable(data, volatile=True)
	if opt.use_gpu:
	input = input.cuda()
	socre=model(input)
	probability = t.nn.functional.softmax(score)[:,1].data.tolist()
	batch_results = [(path_,probability_) \
	        for path_,probability_ in zip(path,probability) ]
        results += batch_results
    write_csv(results,opt.result_file)
    return results


if __name__ == '__main__':
	import fire
	fire.Fire()