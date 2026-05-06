import io
import lmdb
import pickle
from PIL import Image
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
        self.transform = transform

    def __getitem__(self, index):
        with self.env.begin() as txn:
            datum = txn.get(f"{index:08d}".encode())
        obj = pickle.loads(datum)
        img_bytes = obj['img']

        # 使用 PIL 解码
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, obj['label']

    def __len__(self):
        return self.length

class ImageFolderDataset(Dataset):
    def __init__(self, img_list_path, transform=None, target_transform=None):
        file = open(img_list_path, "rb")
        self.img_list = pickle.load(file)
        file.close()
        
        self.length = len(self.img_list)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, target = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

if __name__ == "__main__":
    train_set = ImageFolderDataset("ms1mv3.pickle")