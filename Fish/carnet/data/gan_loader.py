from torch.utils.data import DataLoader
from carnet.data.gan_trajectories import seq_collate_gan, Datasets



def data_loader(args, path, validation):
    dset = Datasets(path, validation)
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate_gan)
    return dset, loader