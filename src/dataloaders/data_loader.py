import pandas as pd

from src.base.base_data_loader import BaseDataLoader
from src.dataloaders.preprocess import get_indices_split
from src.configs.constants import DATA_DIR, DATA_PATH, IMG_COL, CLASS_COL


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        """Initializes data splits and generators from the data file.
        """
        super(DataLoader, self).__init__(config)



        # Get stratified split indices
        train_idx, val_idx = get_indices_split(df, CLASS_COL, 0.2)


    def get_train_gen(self):
        """Retrieves the data generator to be fed to a Keras model.fit_generator.

        Provides batches of images and its corresponding label.
        """
        return self.train_gen


    def get_val_gen(self):
        return self.val_gen


    def get_test_gen(self):
        return self.test_gen


    def get_class_to_label(self):
        return self.class_to_label


    def get_label_to_class(self):
        return self.label_to_class


    def get_val_labels(self):
        return self.val_labels


    def _init_gens(self, classes, train_data, val_data, test_data):
        """Initializes all the generators, i.e. train/val/test,
        and also the label maps.
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
            y_col=CLASS_COL,
            target_size=(IMG_SIZE, IMG_SIZE),
            classes=classes,
            class_mode='sparse',
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=42
        )
        self.val_gen = val_img_gen.flow_from_dataframe(
            val_data,
            directory=DATA_DIR,
            x_col=IMG_COL,
            y_col=CLASS_COL,
            target_size=(IMG_SIZE, IMG_SIZE),
            classes=classes,
            class_mode='sparse',
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
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

        # Store label maps
        self.class_to_label = self.train_gen.class_indices
        self.label_to_class = dict((v,k) for k,v in self.class_to_label.items())
