import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class RoadDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self,opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseDataset.__init__(self,opt)

        if 'txt' not in opt.phase:
            opt.phase = opt.phase + '.txt'
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = open(self.dir_AB).read().split('\n') # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random interger index
        AB_path = self.AB_paths[index].split()
        #print(AB_path)

        A_path = AB_path[0]
        B_path = AB_path[1]
        A = Image.open('./data_road/'+A_path).convert('RGB')
        B = Image.open('./data_road/'+B_path).convert('RGB')


        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)



        return {'A':A, 'B':B, 'A_path':A_path,'B_path':B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


