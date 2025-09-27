# Di bagian data loader ini
# Kita akan mengerjakan untuk membungkus dataset
# Menjadi iterable untuk training loop
# Jadi disini tidak ada proses Tokenisasi seperti dataset
# Jadi inti nya disini hanya lah batching data saja

from torch.utils.data import DataLoader
def membuat_dataloader(dataset=None, file_path=None, batch_size=64, target_vocab_size=1000, seq_len=64, shuffle=True):
    """
    Membuat DataLoader baik dari file_path (TextDataSet baru) atau dataset yang sudah ada.
    """
    if dataset is None:
        if file_path is None:
            raise ValueError("Harus memberi dataset atau file_path")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
