"""
Script to be run when training models, as per config file specified.
"""
import sys

from src.utils.config import process_config
from src.utils.dirs import create_dirs
from src.utils.args import get_args
from src.utils import factory

# Enable eager execution at program startup??
# import tensorflow as tf
# tf.enable_eager_execution()

def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    # create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = factory.create("src.dataloaders."+config.data_loader.name)(config)

    print('Create the model.')
    model = factory.create("src.models."+config.model.name)(config)

    print('Create the trainer')
    trainer = factory.create("src.trainers."+config.trainer.name)(
        model.model,
        data_loader,
        # data_loader.get_train_datagen(),
        # data_loader.get_val_datagen(),
        config,
    )

    # Print model summary just as a sanity check
    model.model.summary()

    print('Start training the model.')
    history = trainer.train()

if __name__ == '__main__':
    main()
