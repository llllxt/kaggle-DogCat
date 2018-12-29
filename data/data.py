from torch.utils import data

class DogCat(data.Dataset):
	def __init__(self, root, transform=None, train=True, test=False):

		self.test = test
		imgs = [os.path.join(root, img) for img in os.listdir(root)]
		if not test:
			imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

		img_num = len(imgs)

		if self.test:
			self.imgs = imgs
		elif train:
			self.imgs = imgs[:int(0.7 * imgs_num)]
		else:
			self.imgs = imgs[int(0.7 * imgs_num):]

		if transforms is None:
			normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
									std = [0.229, 0.224, 0.225])

			if not train:
				self.transforms = T.Compose([
					T.Resize(224),
					T.CenterCrop(224),
					T.ToTensor(),
					normalize
					])
			else:
				self.transforms = T.Compose([
					T.Resize(224),
					T.RandomReSizedCrop(224),
					T.RandomHorizontalFlip(224),
					T.ToTensor(),
					normalize
					])

		def __getitem__(self,index):
			img_path = self.imgs[index]
			if not self.test:
				label = int(self.imgs[index].split('.')[-2].split('')[-1])
			else:
				label = 1 if 'dog' in img_path.split('/')[-1] else 0

			data = Image.open(img_path)
			data = self.transforms(data)
			return data. label

		def __len__(self):
			return len(self.imgs)