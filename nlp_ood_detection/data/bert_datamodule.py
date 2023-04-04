import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from datasets import Dataset

from nlp_ood_detection.data.data_processing import DataPreprocessing


class BertBasedDataModule(pl.LightningDataModule):
    """ DataModule for BERT based models """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 dataset_name: str = 'imdb',
                 batch_size: int = 64,
                 num_workers: int = 1,
                 **kwargs
                 ) -> None:
        super(BertBasedDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.datasets, self.train, self.test, self.val = None, None, None, None

    def setup(self, stage=None):
        self.datasets = DataPreprocessing(
            tokenizer=self.tokenizer, dataset_list=[self.dataset_name])

        self.train = self.datasets[self.dataset_name]['train']
        self.test = self.datasets[self.dataset_name]['test']
        self.val = self.datasets[self.dataset_name]['val']

    def _dataloader(self, dataset: Dataset, shuffle: bool = False):
        """ Create a data loader for a given dataset

        Args:
            dataset (Dataset): Given dataset
            shuffle (bool, optional): Wheter or not to shuffle. Defaults to False.
        """
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def train_dataloader(self):
        return self._dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val)

    def test_dataloader(self):
        return self._dataloader(self.test)
