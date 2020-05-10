# Share-GAN

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/jiaxu0017/Share-GAN
cd Share-GAN
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.
  
  
### share-GAN train/test
- Download a share-GAN dataset (e.g. maps)  Copy the dataset into the project path (*** / ***)
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./***/*** --dataset_mode  road --model branch --preprocess resize --load_size 256 --batch_size 4 --gpu_ids 1,2 --netD full
```
To see more intermediate results, check out `./checkpoints/maps_share-gan/web/index.html`.
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./***/***--dataset_mode  road --model branch --preprocess resize --load_size 256 --batch_size 4
--gpu_ids 1,2
```
- The test results will be saved to a html file here: `./results/maps_share-gan/latest_test/index.html`.
