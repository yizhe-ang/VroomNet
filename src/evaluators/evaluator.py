"""
Evaluates a model based on a fixed dataset.

Class that contains:
    - The Model to evaluate,
    - Data loader

Able to:
    - Evaluate model based on data,
    - Make predictions
    - Show statistics based on model predictions on data,
    - PLOT (images, confusion matrix, etc.)
"""
class Evaluator(object):
    def __init__(self, model, data_loader, config):
        """config.evaluator will contain:

        data: Which part of the dataset to evaluate on.
            i.e. One of ["train", "val", "all"]
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config

        # Initialize performance (loss, acc)
        self.performance = self.model.evaluate_generator(
            self.data_loader.get_val_datagen()
        )
        # Initialize predictions
        self.predictions = self.model.predict_generator(
            self.data_loader.get_val_datagen()
        )
        # Initialize raw data
        self.data = self.data_loader.get_val_data()


    
