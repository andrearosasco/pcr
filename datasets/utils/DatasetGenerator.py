import os
from pathlib import Path

import tqdm


class DatasetGenerator:
    def __init__(self, dataset_path, parts, oversampling=True, classes=None):
        self.data_path = Path(dataset_path)
        self.train_size = parts[0]
        self.val_size = parts[1]
        self.test_size = parts[2]
        self.oversampling = oversampling
        self.classes = classes

        assert round(self.val_size + self.train_size + self.test_size, 5) == 1.

    def generate(self):
        """
        Given a folder with folder classes inside that contains samples, it gets the number of elements in the class
        with more elements and repeat the class with less samples such that each class contains max_length samples
        :return:
        """
        all_classes = list(filter(lambda x: x.is_dir(), self.data_path.glob('*')))

        with (self.data_path / 'classes.txt').open('r') as f:
            ids_names = {l.split()[1]: l.split()[2] for l in f.readlines()}
        names_ids = {v: k for k, v in ids_names.items()}

        lengths = [len(os.listdir(self.data_path / names_ids[cls])) for cls in self.classes]
        max_length = max(lengths)
        max_train = int(max_length * self.train_size)

        train = []
        valid = []
        test = []

        for cls in tqdm.tqdm(all_classes):
            try:
                if self.classes is not None and ids_names[cls.name] not in self.classes:
                    continue
            except KeyError:
                print(f'Directory {cls.name} is not a recognized ShapeNet directory and it will be ignored')
                continue

            # Get samples of that class and divide it into train val and test
            files = list(cls.glob('*'))

            no_train = int(len(files) * self.train_size)
            no_val = int(len(files) * self.val_size)

            train_local = files[:no_train]
            valid_local = files[no_train:(no_train + no_val)]
            test_local = files[(no_train + no_val):]

            if self.oversampling:
                while max_train - len(train_local) >= len(train_local):
                    train_local.extend(train_local)
                train_local.extend(train_local[:max(0, max_train - len(train_local))])

            train.extend(train_local)
            valid.extend(valid_local)
            test.extend(test_local)

        if self.oversampling:
            assert len(train) == max_train * len(self.classes)

        with (self.data_path / "train.txt").open('w') as f:
            lines = [f'{elem.parent.name}/{elem.name}' for elem in train]
            print(*lines, sep='\n', end='', file=f)

        with (self.data_path / "valid.txt").open('w') as f:
            lines = [f'{elem.parent.name}/{elem.name}' for elem in valid]
            print(*lines, sep='\n', end='', file=f)

        with (self.data_path / "test.txt").open('w') as f:
            lines = [f'{elem.parent.name}/{elem.name}' for elem in test]
            print(*lines, sep='\n', end='', file=f)


if __name__ == "__main__":
    from configs import DataConfig
    classes = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel', 'microwave']
    iterator = DatasetGenerator(DataConfig.dataset_path, [0.7, 0.1, 0.2], oversampling=True, classes=classes)
    iterator.generate()
