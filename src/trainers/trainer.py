from src.configs.constants import (
    MODEL_WEIGHTS_DIR, MODEL_LOGS_DIR, MODEL_CSV_DIR, EXPS_PATH
)
from src.base.base_trainer import BaseTrainer

import os
import csv
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
)

class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, config):
        super(Trainer, self).__init__(model, data_loader, config)
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        PATIENCE = self.config.trainer.patience

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(MODEL_WEIGHTS_DIR, f'{self.config.exp.name}.h5'),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=os.path.join(MODEL_LOGS_DIR, self.config.exp.name)
            )
        )

        self.callbacks.append(
            EarlyStopping(
                patience=PATIENCE,
                restore_best_weights=False
            )
        )

        self.callbacks.append(
            CSVLogger(
                filename=os.path.join(MODEL_CSV_DIR, f'{self.config.exp.name}.csv'),
                append=True
            )
        )


    def train(self):
        train_gen = self.data_loader.get_train_gen()
        val_gen = self.data_loader.get_val_gen()

        steps_per_epoch = train_gen.n // train_gen.batch_size
        validation_steps = val_gen.n // val_gen.batch_size

        history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=val_gen,
            validation_steps=validation_steps,
            # use_multiprocessing=True,
            # workers=6
        )

        self._log_exp_info(history)

        return history


    def _log_exp_info(self, history):
        # log metrics like precision and recall too?
        PATIENCE = self.config.trainer.patience

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        with open(EXPS_PATH, mode='a') as exps:
            exps_writer = csv.writer(exps, delimiter=',')
            exps_writer.writerow([
                self.config.exp.name,             # exp_name
                str(self.config.model.toDict()),  # model
                self.config.data_loader.img_size, # img_size
                self.model.count_params(),        # n_params
                len(val_loss) - PATIENCE,         # n_epochs
                min(val_loss),                    # min_loss
                max(acc),                         # max_acc
                loss[-(PATIENCE + 1)],            # train_loss
                acc[-(PATIENCE + 1)],             # train_acc
                val_loss[-(PATIENCE + 1)],        # val_loss
                val_acc[-(PATIENCE + 1)],         # val_acc
                '',                               # remarks
            ])
