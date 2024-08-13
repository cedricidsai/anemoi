import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp

import xskillscore as xs

import xarray as xr
import time
import numpy as np

import os
import sys

from makani.models.networks.sfnonet import SphericalFourierNeuralOperatorNet as SFNO


class XRDataset(torch.utils.data.Dataset):
    def __init__(self, device, dt, xr_data, dims=[], nlat=91, nlon=144, npres=11):
        self.dt = dt
        self.xr_data = xr_data        
        self.device = device
        self.dims=dims
        self.nlat= nlat
        self.nlon = nlon
        self.npres = npres
        self.ncha = self.npres * len(self.dims)
        self.__stats__()

    def __len__(self) -> int:
        return len(self.xr_data.coords['time']) - self.dt

    # Prepare input target pairs as normalised tensors of channels
    def __getitem__(self, idx):
        x = self.normalize(torch.as_tensor(np.array(self.xr_data.isel({'time': idx})[self.dims].to_array()).reshape(-1, self.nlat, self.nlon), device=self.device, dtype=torch.float32))
        y = self.normalize(torch.as_tensor(np.array(self.xr_data.isel({'time' : idx+self.dt})[self.dims].to_array()).reshape(-1, self.nlat, self.nlon), device=self.device, dtype=torch.float32))
        return (x, y)

    # Calculate stats for normalisation
    def __stats__(self):
        dims_means = self.xr_data[self.dims].mean(dim=['time','pressure', 'latitude', 'longitude'])
        dims_std = self.xr_data[self.dims].std(dim=['time','pressure', 'latitude', 'longitude'])
        self.means = torch.as_tensor(np.repeat(np.array(dims_means.to_array()).flatten(), self.npres*self.nlat*self.nlon).reshape(1,self.ncha,self.nlat,self.nlon), device=self.device, dtype=torch.float32)
        self.stds = torch.as_tensor(np.repeat(np.array(dims_std.to_array()).flatten(), self.npres*self.nlat*self.nlon).reshape(1,self.ncha,self.nlat,self.nlon), device=self.device, dtype=torch.float32)

    # Tensor multidimensional channel normalisation
    def normalize(self, tensor):
        return ((tensor - self.means) / self.stds).squeeze()

    # Tensor multidimensional channel denormalisation
    def denormalize(self, tensor):
        return (tensor * self.stds) + self.means

def train_model(model, dataset, dataloader, optimizer, gscaler, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8):

    enable_amp = False

    train_start = time.time()

    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        # do the training
        acc_loss = 0
        model.train()
        for inp, tar in dataloader:
            with amp.autocast(enabled=enable_amp):
                prd = model(inp)

                for _ in range(nfuture):
                    prd = model(prd)
        
                loss = loss_fn(prd, tar)
                acc_loss += loss.item() * inp.size(0)
                optimizer.zero_grad(set_to_none=True)
                # gscaler.scale(loss).backward() #
                loss.backward()
                optimizer.step()
                # gscaler.update() #

        acc_loss = acc_loss / len(dataloader.dataset)

        # perform validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)

                for _ in range(nfuture):
                    prd = model(prd)

                loss = loss_fn(prd, tar)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        print(f'--------------------------------------------------------------------------------', flush=True)
        print(f'Epoch {epoch} summary:', flush=True)
        print(f'time taken: {epoch_time}', flush=True)
        print(f'learning rate {scheduler.get_last_lr()}', flush=True)
        print(f'accumulated training loss: {acc_loss}', flush=True)
        print(f'relative validation loss: {valid_loss}', flush=True)

    train_time = time.time() - train_start

    print(f'--------------------------------------------------------------------------------')
    print(f'done. Training took {train_time}.')
    return valid_loss


def train(training_data_file, dims, filter_type, spectral_transform, grid, epochs, num_layers, scale_factor, embed_dim):
    training_data = xr.load_dataset(training_data_file)

    enable_amp = False

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    print(device, flush=True)

    nlat = len(training_data.coords['latitude'])
    nlon = len(training_data.coords['longitude'])
    npres = len(training_data.coords['pressure'])

    dt = 1

    dataset = XRDataset(device, dt, training_data, dims, nlat, nlon, npres)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False, drop_last=True)

    channels = dataset.ncha

    model = SFNO(spectral_transform=spectral_transform, operator_type='dhconv', inp_shape=(nlat, nlon), out_shape=(nlat, nlon), model_grid_type=grid,
                num_layers=num_layers, scale_factor=scale_factor, inp_chans=channels, out_chans=channels, embed_dim=embed_dim, big_skip=False,
                pos_embed='direct', use_mlp=False, activation_function = "gelu", normalization_layer="layer_norm").to(device)

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=0.0)

    gscaler = amp.GradScaler(enabled=enable_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, mode='min', threshold=0.1)

    train_model(model, dataset, dataloader, optimizer, gscaler, scheduler, nepochs=int(epochs), nfuture=dt-1)

    # save model checkpoint (timestamp + parameters filename)

    model_name = str(int(time.time())) + '.pt'

    root_path = os.path.dirname(__file__)

    torch.save(model.state_dict(), os.path.join(root_path, '../models/' + model_name))

    return model, device, dataset

def finetune(model, device, trainset, nsteps, finetuning_epochs):
    
    # Finetune for stability
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False, drop_last=True)

    for dt in range(2, nsteps):
        trainset.dt = dt
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=0.0)
        gscaler = amp.GradScaler(enabled=False)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=finetuning_epochs)
        train_model(model, trainset, dataloader, optimizer, gscaler, scheduler, nepochs=int(finetuning_epochs), nfuture=dt-1)

    # save model checkpoint (timestamp + parameters filename)

    model_name = str(int(time.time())) + '.pt'

    root_path = os.path.dirname(__file__)

    torch.save(model.state_dict(), os.path.join(root_path, '../models/' + model_name))

    return model, device, trainset


def prepare(training_data_file, dims, filter_type, spectral_transform, grid, epochs, num_layers, scale_factor, embed_dim):

    training_data = xr.load_dataset(training_data_file)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    print(device, flush=True)

    nlat = len(training_data.coords['latitude'])
    nlon = len(training_data.coords['longitude'])
    npres = len(training_data.coords['pressure'])

    dt = 1

    dataset = XRDataset(device, dt, training_data, dims, nlat, nlon, npres)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False, drop_last=True)

    channels = dataset.ncha

    model = SFNO(spectral_transform=spectral_transform, operator_type='dhconv', inp_shape=(nlat, nlon), out_shape=(nlat, nlon), model_grid_type=grid,
                num_layers=num_layers, scale_factor=scale_factor, inp_chans=channels, out_chans=channels, embed_dim=embed_dim, big_skip=False,
                pos_embed='direct', use_mlp=False, activation_function = "gelu", normalization_layer="layer_norm").to(device)

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    return model, device, dataset


def experiment_training(model, device, trainset, nsteps, finetuning_epochs):
    
    dataloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False, drop_last=True)

    for dt in reversed(range(1, nsteps)):
        trainset.dt = dt
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=0.0)
        gscaler = amp.GradScaler(enabled=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, mode='min', threshold=0.1)
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=finetuning_epochs)
        train_model(model, trainset, dataloader, optimizer, gscaler, scheduler, nepochs=int(finetuning_epochs), nfuture=dt-1)

    # save model checkpoint (timestamp + parameters filename)

    model_name = str(int(time.time())) + '.pt'

    root_path = os.path.dirname(__file__)

    torch.save(model.state_dict(), os.path.join(root_path, '../models/' + model_name))

    return model, device, trainset


def load(model_file_name, sfno_lib, training_data_file, dims, filter_type, spectral_transform, grid, epochs, num_layers, scale_factor, embed_dim):

    training_data = xr.load_dataset(training_data_file)

    enable_amp = False

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    print(device, flush=True)

    nlat = len(training_data.coords['latitude'])
    nlon = len(training_data.coords['longitude'])
    npres = len(training_data.coords['pressure'])

    dataset = XRDataset(device, 1, training_data, dims, nlat, nlon, npres)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False, drop_last=True)

    channels = dataset.ncha

    model = SFNO(spectral_transform=spectral_transform, operator_type='dhconv', inp_shape=(nlat, nlon), out_shape=(nlat, nlon), model_grid_type=grid,
                num_layers=num_layers, scale_factor=scale_factor, inp_chans=channels, out_chans=channels, embed_dim=embed_dim, big_skip=False,
                pos_embed='direct', use_mlp=False, activation_function = "gelu", normalization_layer="layer_norm").to(device)

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    root_path = os.path.dirname(__file__)

    ckpt = torch.load(os.path.join(root_path, '../models/' + model_file_name))
    model.load_state_dict(ckpt)

    return model, device, dataset

def test(model, device, test_data_file, dims, trainset):

    test_data = xr.load_dataset(test_data_file)

    nlat = len(test_data.coords['latitude'])
    nlon = len(test_data.coords['longitude'])
    npres = len(test_data.coords['pressure'])

    dataset = XRDataset(device, 1, test_data, dims, nlat, nlon, npres)

    dataset.means = trainset.means
    dataset.stds = trainset.stds

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, drop_last=True)

    print(' Test set result')

    with torch.inference_mode():
        inp, tar = next(iter(dataloader))
        prd = model(inp)

        ytar = xr.DataArray(tar.detach().cpu().numpy(), dims=['b', 'v','lat','lon'])[0,:,:,:]
        yprd = xr.DataArray(prd.detach().cpu().numpy(), dims=['b', 'v','lat','lon'])[0,:,:,:]

        print(xs.mse(ytar, yprd, dim=['lat','lon']))

        print(ytar[0,:,:])
        print(yprd[0,:,:])

        print(dataset.denormalize(tar))
        print(dataset.denormalize(prd))


def rollout(model, device, test_data_file, dims, nfuture, trainset):
    # run auto regression n starts
        
    test_data = xr.load_dataset(test_data_file)

    nlat = len(test_data.coords['latitude'])
    nlon = len(test_data.coords['longitude'])
    npres = len(test_data.coords['pressure'])

    dataset = XRDataset(device, 1, test_data, dims, nlat, nlon, npres)

    dataset.means = trainset.means
    dataset.stds = trainset.stds
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, persistent_workers=False, drop_last=True)

    print(' Test set rollout')

    rollout = xr.DataArray()

    with torch.inference_mode():

        inp, tar = next(iter(dataloader))
        prd = None
        for i in range(nfuture):
            if i == 0:
                prd = model(inp)
            else:
                prd = model(prd)

            ytar = xr.DataArray(tar.detach().cpu().numpy(), dims=['time','v','lat','lon'])[0,:,:,:]
            yprd = xr.DataArray(dataset.denormalize(prd).detach().cpu().numpy(), dims=['time','v','lat','lon'])[0,:,:,:]

            rollout = xr.concat([rollout, yprd], dim='time')

    name = str(int(time.time())) + '.nc'

    rollout.to_netcdf('../rollouts/' + name)

            # print(xs.mse(ytar, yprd, dim=['lat','lon']))

            # print(ytar[0,:,:])
            # print(yprd[0,:,:])
            # y = xr.DataArray(prd-tar, dims=['b','v','lat','lon'])[0,1,:,:].plot()

            # print(prd)
            # y = xr.DataArray(prd-tar, dims=['b','v','lat','lon'])[0,1,:,:].plot()
            # y = xr.DataArray(tar, dims=['b','v','lat','lon'])[0,0,:,:].plot()
            # print(xr.DataArray(prd-tar, dims=['b','v','lat','lon']))#.plot()




if __name__=="__main__":
    filter_type = sys.argv[1]
    training_data_file=sys.argv[2]
    test_data_file=sys.argv[3]
    num_layers = int(sys.argv[4])
    embed_dim = int(sys.argv[5])
    nsteps = int(sys.argv[6])
    epochs = int(sys.argv[7])
    dims = sys.argv[8].split(',')

    # forcings = 

    # model, device, dataset = train(sfno_lib=sfno_lib, training_data_file=training_data_file, dims=dims, filter_type=filter_type, spectral_transform='sht', grid='equiangular', epochs = epochs, num_layers=num_layers, scale_factor=1, embed_dim=512)

    # model, device, dataset = finetune(model=model, device=device, trainset=dataset, nsteps=5, finetuning_epochs=int(epochs/2))

    model, device, dataset = prepare(training_data_file=training_data_file, dims=dims, filter_type=filter_type, spectral_transform='sht', grid='equiangular', 
                                     epochs = epochs, num_layers=num_layers, scale_factor=1, embed_dim=embed_dim)

    model, device, dataset = experiment_training(model=model, device=device, trainset=dataset, nsteps=nsteps, finetuning_epochs=epochs)

    print('experiment : ', sys.argv[0])

    print('parameters : ', filter_type, num_layers, epochs)

    test(model, device, test_data_file=test_data_file, dims=dims, trainset=dataset)
    
    rollout(model, device, test_data_file=test_data_file, dims=dims, nfuture=31*4, trainset=dataset)

    # main(training_data_file=sys.argv[1], test_data_file=sys.argv[2], spectral_transform='fft' , grid="none", epochs = 1, num_layers=10, scale_factor=3, embed_dim=256)
    # main(training_data_file=sys.argv[1], test_data_file=sys.argv[2], spectral_transform=sys.argv[3] ,epochs = sys.argv[3], num_layers=sys.argv[4], scale_factor=sys.argv[5], embed_dim=sys.argv[6])
