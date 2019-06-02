import os
import csv
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
)
from sklearn.metrics import precision_recall_fscore_support

from src.configs.constants import (
    MODEL_WEIGHTS_DIR, MODEL_LOGS_DIR, MODEL_CSV_DIR, EXPS_PATH
)
from src.base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, data_loader, config):
        super(Trainer, self).__init__(model, data_loader, config)
        self.callbacks = []
        self._init_callbacks()


    def train(self):
        N_EPOCHS = self.config.trainer.n_epochs

        train_gen = self.data_loader.get_train_gen()
        val_gen = self.data_loader.get_val_gen()

        steps_per_epoch = train_gen.n // train_gen.batch_size
        validation_steps = val_gen.n // val_gen.batch_size

        history = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=N_EPOCHS,
            verbose=1,
            callbacks=self.callbacks,
            validation_data=val_gen,
            validation_steps=validation_steps,
            # use_multiprocessing=True,
            # workers=6
        )

        self._log_exp_info(history)

        return history


    def _init_callbacks(self):
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

        # Only perform early stopping if specified
        if self.config.trainer.early_stopping:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.trainer.patience,
                    # Restore best weights for evaluation at the end of training.
                    restore_best_weights=True
                )
            )

        self.callbacks.append(
            CSVLogger(
                filename=os.path.join(MODEL_CSV_DIR, f'{self.config.exp.name}.csv'),
                append=True
            )
        )


    def _log_exp_info(self, history):
        """Retrieves any metrics and logs them for this experiment.
        """
        if self.config.trainer.early_stopping:
            PATIENCE = self.config.trainer.patience
        else:
            PATIENCE = 0

        acc = history.history['acc']
        val_acc = history.history['val_acc']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        precision, recall, fscore = self._eval_model()

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
                precision,                        # precision
                recall,                           # recall
                fscore,                           # fscore
                '',                               # remarks
            ])


    def _eval_model(self):
        """Computes the precision, recall, and fscore of the model
        at the end of training.
        """
        val_gen = self.data_loader.get_val_gen()
        y_true = self.data_loader.get_val_labels()
        predictions = self.model.predict_generator(val_gen)

        y_pred = np.argmax(predictions, axis=1)

        precision, recall, fscore, _ = \
            precision_recall_fscore_support(y_true, y_pred, average='micro')

        return precision, recall, fscore
