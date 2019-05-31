"""
Contains any functions relating to evaluation and the Evaluator object (plotting, calculating statistics etc.)
"""
from src.utils import factory

# Creates the Evaluator object from the JSON config file path.
def create_evaluator(json_file):
    """Takes in the path of the JSON config file.

    Returns the Evaluator object created.
    """
    config = process_config(args.config)

    # Initialize the Data Loader
    data_loader = factory.create("src.dataloader."+config.data_loader.name)(config)
    # Initialize the Model (specifying weights)
    model = factory.create("src.models."+config.model.name)(config)
    # Initialize the Evaluator
    evaluator = factor.create("src.evaluators."+config.evaluator.name)(
        model.model,
        data_loader,
        config,
    )

    return evaluator
