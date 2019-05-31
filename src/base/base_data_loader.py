class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_datagen(self):
        raise NotImplementedError

    def get_test_datagen(self):
        raise NotImplementedError
