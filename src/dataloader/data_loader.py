from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.base.base_data_loader import BaseDataLoader


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        
        # Generate Train/Val split indices
        self.train_indices, self.val_indices = get_indices_split(self.data)

        # Initialize ImageDataGenerators (also comprises augmentations)
        train_datagen = 
        val_datagen = 

        # Perform any preprocessing
        self.data = self._preprocess(DATA_PATH)


    def _preprocess(self, data_path):
        """Performs any preprocessing of the data file specified.
        """
        return process_via_annotations(data_path)

    def get_train_datagen(self):
        """Creates a data generator to be fed to a Keras model.fit_generator.

        Provides batches of images and its corresponding label.
        """
        img_dim = self.config.data_loader.img_dim

        return ImageDataGenerator(
            self.train_indices,
            self.data,
            (img_dim, img_dim, 3),
            IMG_DIR,
            aug=self.aug,
            batch_size=25
        )

    def get_val_datagen(self):
        img_dim = self.config.data_loader.img_dim

        return ImageDataGenerator(
            self.val_indices,
            self.data,
            (img_dim, img_dim, 3),
            IMG_DIR,
            aug=None,
            batch_size=29,
            shuffle=False
        )

    def get_val_data(self):
        """Returns raw data (without processing image and labels).
        """
        return self.data.loc[self.val_indices]
