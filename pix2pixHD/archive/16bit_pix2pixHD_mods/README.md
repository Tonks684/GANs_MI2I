# Pix2PixHD 16bit hacks

./my_train.py

	- Line 12:
		#from data.data_loader import CreateDataLoader
		from data.my_data_loader import CreateDataLoader

		#import util.util as util
		import util.my_util as util

		from util.my_visualizer import Visualizer
		#from util.visualizer import Visualizer

./utils/my_visualizer.py

	- Line 5:
		#from . import util
		from . import my_util as util

	- Line 58:
		if self.use_html: # save images to a html file
			for label, image_numpy in visuals.items():
				if isinstance(image_numpy, list):
					for i in range(len(image_numpy)):
						#img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.jpg' % (epoch, label, i))
						img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%d.tiff' % (epoch, label, i))
						util.save_image(image_numpy[i], img_path)
				else:
					#img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
					img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.tiff' % (epoch, label))
					util.save_image(image_numpy, img_path)

./utils/my_util.py

	- Line 10:
		#def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
		def tensor2im(image_tensor, imtype=np.uint16, normalize=True):
		if isinstance(image_tensor, list):
			image_numpy = []
			for i in range(len(image_tensor)):
				image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
			return image_numpy
		image_numpy = image_tensor.cpu().float().numpy()
		if normalize:
			#image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
			image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 65535.0
		else:
			#image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
			image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 65535.0
		#image_numpy = np.clip(image_numpy, 0, 255)
		image_numpy = np.clip(image_numpy, 0, 65535)
		if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
			image_numpy = image_numpy[:,:,0]
		return image_numpy.astype(imtype)

	- Line 27:
		#def tensor2label(label_tensor, n_label, imtype=np.uint8):
		def tensor2label(label_tensor, n_label, imtype=np.uint16):

./data/my_dataloader.py

	def CreateDataLoader(opt):
		#from data.custom_dataset_data_loader import CustomDatasetDataLoader
		from data.my_custom_dataset_data_loader import CustomDatasetDataLoader
		data_loader = CustomDatasetDataLoader()
		print(data_loader.name())
		data_loader.initialize(opt)
		return data_loader

./data/my_custom_dataset_data_loader.py

	def CreateDataset(opt):
		dataset = None
		#from data.aligned_dataset import AlignedDataset
		from data.my_aligned_dataset import AlignedDataset
		dataset = AlignedDataset()

		print("dataset [%s] was created" % (dataset.name()))
		dataset.initialize(opt)
		return dataset

./data/my_aligned_dataset.py

	- Line 2:
		#from data.base_dataset import BaseDataset, get_params, get_transform, normalize
		from data.my_base_dataset import BaseDataset, get_params, get_transform, normalize

	- Line 38:
		def __getitem__(self, index):
			### input A (label maps)
			A_path = self.A_paths[index]
			A = Image.open(A_path)
			params = get_params(self.opt, A.size)
			if self.opt.label_nc == 0:
				transform_A = get_transform(self.opt, params)
				#A_tensor = transform_A(A.convert('RGB')
				A_tensor = transform_A(A.convert('F').point(lambda p: p*(1/65535)))
			else:
				transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
				A_tensor = transform_A(A) * 255.0

			B_tensor = inst_tensor = feat_tensor = 0
			### input B (real images)
			if self.opt.isTrain or self.opt.use_encoded_image:
				B_path = self.B_paths[index]
				#B = Image.open(B_path).convert('RGB')
				B = Image.open(B_path).convert('F')
				transform_B = get_transform(self.opt, params)
				#B_tensor = transform_B(B)
				B_tensor = transform_B(B.point(lambda p: p*(1/65535)))

./data/my_base_dataset.py

	- Line 55:
		if normalize:
			#transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
			#                                        (0.5, 0.5, 0.5))]
			transform_list += [transforms.Normalize((0.5), (0.5))]

		return transforms.Compose(transform_list)

	- Line 61:
		def normalize():
			#return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			return transforms.Normalize((0.5), (0.5))



**Inference: test_16bit.py**

    -import util.util_16bit as util
    -from util.visualizer_16bit import Visualizer
    - from util_16bit import html
