import abc
import argparse
import csv
import glob
import os.path
import shutil

from daaja.augmentors.sentence.randam_delete_augmentor import RandamDeleteAugmentor
from daaja.augmentors.sentence.randam_insert_augmentor import RandamInsertAugmentor
from daaja.augmentors.sentence.randam_swap_augmentor import RandamSwapAugmentor
from daaja.augmentors.sentence.synonym_replace_augmentor import SynonymReplaceAugmentor


class IAugmentor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def augment(self, text: str) -> str:
        return NotImplementedError()

    def new_filename(self, filename: str) -> str:
        return NotImplementedError()


class SynonymAug(IAugmentor):
    def __init__(self):
        self.augmenter = SynonymReplaceAugmentor(alpha=0.02)

    def augment(self, text: str) -> str:
        return self.augmenter.augment(text)

    def new_filename(self, filename: str) -> str:
        return "aug_syno_" + filename


class RandomInsertAug(IAugmentor):
    def __init__(self):
        self.augmentor = RandamInsertAugmentor(alpha=0.1)

    def augment(self, text: str) -> str:
        return self.augmentor.augment(text)

    def new_filename(self, filename: str) -> str:
        return "aug_random_insert_" + filename


class RandomDeleteAug(IAugmentor):
    def __init__(self):
        self.augmenter = RandamDeleteAugmentor(p=0.1)

    def augment(self, text: str) -> str:
        return self.augmenter.augment(text)

    def new_filename(self, filename: str) -> str:
        return "aug_random_delete_" + filename


class RandomSwapAug(IAugmentor):
    def __init__(self):
        self.augmenter = RandamSwapAugmentor(alpha=0.1)

    def augment(self, text: str) -> str:
        return self.augmenter.augment(text)

    def new_filename(self, filename: str) -> str:
        return "aug_random_swap_" + filename


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="augmentationを行うためのツール")

    p.add_argument("ratio")
    args = p.parse_args()
    args.ratio = float(args.ratio)
    if args.ratio < 0 or 1 < args.ratio:
        raise Exception("raise is invalid value. It must more than 0 and or less 1.")

    ORIGINAL_DATA = os.environ.get("ORIGINAL_DATA")
    if ORIGINAL_DATA is None:
        raise Exception("invalid environment value. ORIGINAL_DATA is None.")

    AUGMENTED_DATA = os.environ.get("AUGMENTED_DATA")
    if AUGMENTED_DATA is None:
        raise Exception("invalid environment value. AUGMENTED_DATA is None.")

    PLANE_DATA = os.environ.get("PLANE_DATA")
    if PLANE_DATA is None:
        raise Exception("invalid environment value. PLANE_DATA is None.")

    augmenters = [SynonymAug(), RandomDeleteAug(), RandomSwapAug()]
    files = glob.glob(os.path.join(ORIGINAL_DATA, "*.csv"))
    print(files)

    current_num = 0
    aug_num = len(files) * args.ratio

    os.makedirs(AUGMENTED_DATA, exist_ok=True)

    for file in files:
        if current_num < aug_num:
            f = open(file, "r")
            for augmentor in augmentors:
                r = csv.reader(f, delimiter=",")
                # remove header
                next(r)
                aug_filename = augmentor.new_filename(os.path.basename(file))
                aug_file = open(os.path.join(AUGMENTED_DATA, aug_filename), "w")
                for row in r:
                    new_row = row
                    new_row[1] = augmentor.augment(new_row[1])
                    w = csv.writer(aug_file)
                    w.writerow(new_row)
                aug_file.close()
                f.seek(0)
            f.close()
            current_num += 1
            shutil.copy(file, AUGMENTED_DATA)
        else:
            shutil.copy(file, PLANE_DATA)
