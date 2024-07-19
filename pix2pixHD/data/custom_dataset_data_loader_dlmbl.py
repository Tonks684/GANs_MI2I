import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    """
    Create a dataset object based on the given options.

    Args:
        opt (argparse.Namespace): The options for creating the dataset.

    Returns:
        dataset (object): The created dataset object.

    """
    dataset = None
    from data.aligned_dataset_dlmbl import AlignedDataset
    dataset = AlignedDataset()

    print('dataset [%s] was created' % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    """
    A custom data loader for HEKCells dataset.
    """

    def name(self):
        return 'HEKCells Data Loader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
