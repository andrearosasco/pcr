import os
import tqdm


class DatasetIterator:
    def __init__(self, data_path, train_val_test_ratio):
        self.data_path = data_path
        self.train_size = train_val_test_ratio[0]
        self.val_size = train_val_test_ratio[1]
        self.test_size = train_val_test_ratio[2]

    def generate_train_val_test_txt(self):
        """
        Given a folder with folder classes inside that contains samples, it gets the number of elements in the class
        with more elements and repeat the class with less samples such that each class contains max_length samples
        :return:
        """
        classes = os.listdir(self.data_path)

        print("Found " + str(len(classes)) + " classes")

        classes_length = {}
        for c in classes:
            classes_length[c] = len(os.listdir(self.data_path + os.sep + c))

        print(classes_length)

        max_length = max(classes_length.values())
        max_train = int(max_length * self.train_size)
        max_val = int(max_length * self.val_size)
        max_test = int(max_length * self.test_size)

        print("maximum number of elements in train: ", max_train, ", val: ", max_val, " and test:", max_test)
        print("Expected length -> train: ", max_train*len(classes),
              ", val: ", max_val*len(classes),
              ", test: ", max_test*len(classes))

        train = []
        val = []
        test = []

        for c in tqdm.tqdm(classes):
            # Get samples of that class and divide it into train val and test
            files = os.listdir(self.data_path + os.sep + c)
            for_train = int(len(files) * self.train_size)
            for_val = int(len(files) * self.val_size)
            train_local = files[:for_train]
            val_local = files[for_train:(for_train + for_val)]
            test_local = files[(for_train + for_val):]
            # Train
            repeated_files = []
            while len(repeated_files) + len(train_local) < max_train:
                repeated_files += list(c + os.sep + file for file in train_local)
            repeated_files += list(c + os.sep + file for file in train_local[:abs(max_train - len(repeated_files))])
            train += repeated_files
            # Validation
            repeated_files = []
            while len(repeated_files) + len(val_local) < max_val:
                repeated_files += list(c + os.sep + file for file in val_local)
            repeated_files += list(c + os.sep + file for file in val_local[:abs(max_val - len(repeated_files))])
            val += repeated_files
            # Test
            repeated_files = []
            while len(repeated_files) + len(test_local) < max_test:
                repeated_files += list(c + os.sep + file for file in test_local)
            repeated_files += list(c + os.sep + file for file in test_local[:abs(max_test - len(repeated_files))])
            test += repeated_files

        assert len(train) == max_train*len(classes)
        assert len(val) == max_val*len(classes)
        assert len(test) == max_test*len(classes)

        with open("train.txt", "w") as outfile:
            for elem in train:
                outfile.write(elem + '\n')

        with open("val.txt", "w") as outfile:
            for elem in val:
                outfile.write(elem + '\n')

        with open("test.txt", "w") as outfile:
            for elem in test:
                outfile.write(elem + '\n')


if __name__ == "__main__":
    from configs.cfg1 import DataConfig
    iterator = DatasetIterator(DataConfig.prep_path, [0.7, 0.2, 0.1])
    iterator.generate_train_val_test_txt()
