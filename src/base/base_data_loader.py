class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_gen(self):
        raise NotImplementedError

    def get_val_gen(self):
        raise NotImplementedError

    def get_test_gen(self):
        raise NotImplementedError
