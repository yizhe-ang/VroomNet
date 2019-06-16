"""Primary script to run that runs the model on the test dataset
and saves the predictions.

Make sure all images are in the 'data/test' folder.
Predictions will be saved in the root directory as a csv file.
"""
import os
import argparse

import numpy as np
import pandas as pd
from fastai.vision import ImageList
from fastai.basic_data import DatasetType
from fastai.basic_train import load_learner

from src.configs.constants import (
    DATA_DIR, TEST_FOLDER, SAVED_DIR
)
from src.models.ensemble import Ensemble


def main(ensemble, tta, output):
    # Read in test data images from the 'test/' folder
    test_imgs = ImageList.from_folder(
        path=os.path.join(DATA_DIR, TEST_FOLDER),
    )

    # Get predictions
    if ensemble:
        # Load ensemble of learners
        learners = []
        learner_names = ['dpn92', 'inceptionv4', 'se_resnext101']
        for name in learner_names:
            learn = load_learner(
                SAVED_DIR,
                f'{name}.pkl',
                test=test_imgs
            )
            learners.append(learn)

        # Init ensemble
        ensemble = Ensemble(learners)

        # Get predictions
        preds = ensemble.predict(tta)

        # Get classes list
        classes = learners[0].data.classes
        # Get image names list
        img_names = [i.name for i in learners[0].data.test_ds.items]

    else:
        learner_name = 'se_resnext101'

        # Initialize Learner
        learn = load_learner(
            SAVED_DIR,
            f'{learner_name}.pkl',
            test=test_imgs
        )

        # Get predictions
        if tta:
            preds, _ = learn.TTA(ds_type=DatasetType.Test)
        else:
            preds, _ = learn.get_preds(ds_type=DatasetType.Test)

        # Get classes list
        classes = learn.data.classes
        # Get image names list
        img_names = [i.name for i in learn.data.test_ds.items]


    # Initialize DataFrame with the predictions
    df = pd.DataFrame(np.array(preds), columns=classes)
    # Insert image names to DataFrame
    df.insert(0, 'img_name', img_names)

    # Save predictions as csv file
    df.to_csv(output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Performs model inference on test dataset."
    )

    parser.add_argument('-e', '--ensemble', action='store_true',
                        help='Whether to perform ensembling.')
    parser.add_argument('-t', '--tta', action='store_true',
                        help='Whether to perform test-time augmentation.')
    parser.add_argument('-o', '--output', default='predictions.csv',
                        help='Can choose to specify output path.')
                        
    args = parser.parse_args()

    main(args.ensemble, args.tta, args.output)
