import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.base.base_data_loader import BaseDataLoader
from src.dataloaders.preprocess import get_indices_split
from src.configs.constants import DATA_DIR, DATA_PATH, IMG_COL, LABEL_COL


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        """Initializes data splits and generators from the data file.
        """
        super(DataLoader, self).__init__(config)

        data = pd.read_csv(DATA_PATH)
        # Separate Training and Test sets
        test_data = data[data['test'] == 1]
        training_data = data[data['test'] == 0]

        # Generate Train/Val split indices
        train_indices, val_indices = get_indices_split(training_data, LABEL_COL, 0.2)

        # iloc!!!
        val_data = training_data.iloc[val_indices]
        train_data = training_data.iloc[train_indices]

        # Initialize data generators, and the label map
        self._init_gens(train_data, val_data, test_data)


    def get_train_gen(self):
        """Retrieves the data generator to be fed to a Keras model.fit_generator.

        Provides batches of images and its corresponding label.
        """
        return self.train_gen


    def get_val_gen(self):
        return self.val_gen


    def get_test_gen(self):
        return self.test_gen

    def get_label_map(self):
        return self.label_map


    def _init_gens(self, train_data, val_data, test_data):
        """Initializes all the generators, i.e. train/val/test,
        and also the label map.
        """
        IMG_SIZE = self.config.data_loader.img_size
        BATCH_SIZE = self.config.data_loader.batch_size
        VAL_BATCH_SIZE = self.config.data_loader.val_batch_size
        TEST_BATCH_SIZE = self.config.data_loader.test_batch_size

        # Initialize ImageDataGenerators (also comprises augmentations)
        # Utilize preprocessing function from ResNet50
        train_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_gen = train_img_gen.flow_from_dataframe(
            train_data,
            directory=DATA_DIR,
            x_col=IMG_COL,
            y_col=LABEL_COL,
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode='sparse',
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=42
        )
        self.val_gen = val_img_gen.flow_from_dataframe(
            val_data,
            directory=DATA_DIR,
            x_col=IMG_COL,
            y_col=LABEL_COL,
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode='sparse',
            batch_size=VAL_BATCH_SIZE,
            shuffle=True,
            seed=42
        )
        self.test_gen = test_img_gen.flow_from_dataframe(
            test_data,
            directory=DATA_DIR,
            x_col=IMG_COL,
            y_col=None,
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode=None,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False
        )

        class_to_label = self.train_gen.class_indices
        label_to_class = dict((v,k) for k,v in class_to_label.items())

        self.label_map = label_to_class
