class BaseTrainer(object):
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        # self.train_datagen = train_datagen
        # self.val_datagen = val_datagen
        self.config = config

    def train(self):
        raise NotImplementedError
