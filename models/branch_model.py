import torch
from models.base_model import BaseModel
from models import networks
from options.train_options import TrainOptions
from data import create_dataset
from evaluation import culc_acc
from util import util
import matplotlib.pyplot as plt

class BranchModel(BaseModel):
    @ staticmethod
    def modify_commandline_options(parser, is_train = True):
        parser.set_defaults(norm='batch',dataset_mode = 'road')
        if is_train:
            parser.set_defaults(pool_size = 0, gan_model = 'vanilla')
            parser.add_argument('--lambda_L1', type=float, default= 100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """
        Initialize the Branch class
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1','D_real','D_fake','D_origin']
        self.loss_names = ['G_GAN', 'G_L1','D_real','D_fake']
        # specity the models you want to save/dispaly. The training/ test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_B', 'fake_B','mask_B','mask_fake_B']
        # specity the models you want to save to the disk. The training/ test scripts will call <BaseModel.save_network> and <BaseModel.load_network>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else: # during test time, only load G
            self.model_names = ['G']

        self.gpu_ids = opt.gpu_ids
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(1536, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # define loss function
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizersl schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

               Parameters:
                   input (dict): include the data itself and its metadata information.

               The option 'direction' can be used to swap images in domain A and domain B.
               """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.mask_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_path' if AtoB else 'B_path']
        self.A_path = input['A_path' if AtoB else 'B_path']
        self.B_path = input['B_path' if AtoB else 'A_path']

        real_A = self.real_A
        self.real_B = self.swap_channal(real_A, self.mask_B)


    def forward(self):
        # origin
        self.mask_fake_B, self.feature_origin = self.netG(self.real_A)
        real_A_1 = self.real_A
        self.fake_B = self.swap_channal(real_A_1, self.mask_fake_B)

        # real_B = self.real_B
        # real_B = real_B.detach().cpu().numpy()
        # B_show = real_B.transpose([0, 2, 3, 1])
        #
        # plt.figure(figsize=(20, 20))
        # plt.imshow(B_show[1])
        # plt.show()

        #print('*******',self.fake_B.shape)
        # self.batch = self.feature_origin.shape[0]
        #self.feature_origin = self.feature_origin.reshape((self.batch,1536))
        # real_seg
        _, self.feature_real = self.netG(self.real_B)
        # self.batch = self.feature_real.shape[0]
        #self.feature_real = self.feature_real.reshape((self.batch,1536))

        # fake_Seg
        _, self.feature_fake = self.netG(self.fake_B)
        # self.batch = self.feature_fake.shape[0]
        #self.feature_fake = self.feature_fake.reshape((self.batch, 1536))

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""

        # origin
        # pred_origin = self.netD(self.feature_origin.detach())
        # self.loss_D_origin = self.criterionGAN(pred_origin,'origin')
        # real
        pred_real = self.netD(self.feature_real.detach())
        # self.loss_D_real = self.criterionGAN(pred_real, 'real')
        self.loss_D_real = self.criterionGAN(pred_real, True)
        #fake
        pred_fake = self.netD(self.feature_fake.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, 'fake')
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # combine loss asn calculate gradients
        # self.loss_D = (self.loss_D_origin + self.loss_D_real + self.loss_D_fake) / 3.0
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2.0

        # print(self.loss_D)
        return  self.loss_D.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.feature_fake.detach())
        self.loss_G_GAN = self.criterionGAN(pred_fake,True)
        # self.loss_G_GAN = self.criterionGAN(pred_fake,'real')
        # print(self.loss_G_GAN)
        # self.loss_G_GAN.backward()

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 / 2
        # combine loss and calculate gradients

        self.loss_G_L1_2 = self.criterionL1(self.mask_fake_B,self.mask_B) * self.opt.lambda_L1
        #self.loss_G_L1_2 = self.criterionL1(self.feature_fake,self.feature_real) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L1_2
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()


        # updata D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # updata G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        image_B = util.tensor2im(self.mask_B)
        image_output = util.tensor2im(self.mask_fake_B)
        TP, FP, FN, TN = culc_acc(image_B, image_output)
        self.acc_value = TP / (TP + FP + FN)




    def swap_channal(self, origin, mask, channal=2):
        B,S,H,W = origin.shape
        origin_swap = torch.zeros([B,S,H,W]).float().to(self.gpu_ids[0])
        for i in range(3):
            if i != channal:
                origin_swap[:,i,:,:] = origin[:,i,:,:]
            else:
                origin_swap[:, channal, :, :] = mask[:, channal, :, :]

        # origin1 = origin
        # origin1 = origin1.cpu().numpy()
        # print(type(origin1))
        # im_show = origin1.transpose([0, 2, 3, 1])
        #
        # plt.figure(figsize=(20,20))
        # plt.imshow(im_show[1])
        # plt.show()
        return origin_swap


# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
#     dataset_size = len(dataset)    # get the number of images in the dataset.
#     model = BranchModel(opt)
#     for i, data in enumerate(dataset):
#         model.set_input(data)
#         model.optimize_parameters()