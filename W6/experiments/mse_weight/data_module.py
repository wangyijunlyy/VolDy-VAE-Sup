from torch.utils.data import DataLoader
from probts.data import ProbTSDataModule


class JointDataModule(ProbTSDataModule):
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.test_batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, num_workers=0)
