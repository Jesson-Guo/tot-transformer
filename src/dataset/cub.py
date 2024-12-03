import os
import os.path
from typing import Callable, Optional, Tuple, Any
from PIL import Image
from torchvision.datasets.utils import download_url, extract_archive, verify_str_arg
from torchvision.datasets import VisionDataset


class CUB(VisionDataset):
    """
    CUB-200-2011 Dataset.

    Args:
        root (string): Root directory of dataset where ``cub`` folder exists or will be saved to if download is set to True.
        split (string, optional): One of {'train', 'test'}. Specifies the dataset split.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    # URL to download the dataset
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(CUB, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = verify_str_arg(split, "split", ("train", "test"))
        self.base_folder = os.path.join(root, "CUB_200_2011")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it.")

        # Paths to necessary files
        images_txt = os.path.join(self.base_folder, "images.txt")
        labels_txt = os.path.join(self.base_folder, "image_class_labels.txt")
        train_test_split_txt = os.path.join(self.base_folder, "train_test_split.txt")
        classes_txt = os.path.join(self.base_folder, "classes.txt")

        # Read image filenames
        with open(images_txt, "r") as f:
            img_name_list = [line.strip().split(" ")[1] for line in f.readlines()]

        # Read labels and adjust to zero-based indexing
        with open(labels_txt, "r") as f:
            label_list = [int(line.strip().split(" ")[1]) - 1 for line in f.readlines()]

        # Read train/test split
        with open(train_test_split_txt, "r") as f:
            train_test_list = [int(line.strip().split(" ")[1]) for line in f.readlines()]

        # Filter images and labels based on split
        if self.split == "train":
            file_list = [x for i, x in zip(train_test_list, img_name_list) if i == 1]
            self.labels = [x for i, x in zip(train_test_list, label_list) if i == 1]
        else:
            file_list = [x for i, x in zip(train_test_list, img_name_list) if i == 0]
            self.labels = [x for i, x in zip(train_test_list, label_list) if i == 0]

        # Full paths to images
        self.imgs = [os.path.join(self.base_folder, "images", f) for f in file_list]

        # Read class names
        with open(classes_txt, "r") as f:
            self.classes = [line.strip().split(" ")[1].split(".")[1] for line in f.readlines()]

        # Create class_to_idx mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Prepare samples
        self._samples = [(self.imgs[i], self.labels[i]) for i in range(len(self.imgs))]

    def _check_integrity(self) -> bool:
        """Check if dataset is already downloaded and verified."""
        if not os.path.isdir(self.base_folder):
            return False
        # You can add more integrity checks if necessary
        return True

    def download(self) -> None:
        """Download the CUB-200-2011 dataset if it doesn't exist already."""
        if self._check_integrity():
            print("Files already downloaded and verified.")
            return

        os.makedirs(self.base_folder, exist_ok=True)

        # Download the dataset
        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # Extract the dataset
        print("Extracting files...")
        extract_archive(os.path.join(self.root, self.filename), self.root)

        # Verify extraction
        if not self._check_integrity():
            raise RuntimeError("Dataset download or extraction failed.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the class index of the target class.
        """
        path, target = self._samples[index]

        # Load image
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Apply transforms if any
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def extra_repr(self) -> str:
        return f"Split: {self.split}\nNumber of classes: {len(self.classes)}"
