"""
Modul DataLoader untuk Pemrosesan Batch Dataset Language Model.
"""

from torch.utils.data import DataLoader, Dataset


def membuat_dataloader(dataset: Dataset = None,
                       batch_size: int = 64,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = False) -> DataLoader:
    """
    Membuat PyTorch DataLoader dari instance Dataset yang diberikan.
    """
    if dataset is None:
        raise ValueError("Parameter 'dataset' tidak boleh None.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
