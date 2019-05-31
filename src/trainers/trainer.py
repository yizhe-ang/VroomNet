from src.configs.constants import (
    MODEL_WEIGHTS_DIR, MODEL_LOGS_DIR, MODEL_CSV_DIR, EXPS_PATH
)
from src.base.base_trainer import BaseTrainer

import os
import csv
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
)

class ModelTrainer(BaseTrainer):
    def __init__(self, model, data_loader, config):
        super(ModelTrainer, self).__init__(model, data_loader, config)
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
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
                patience=20,
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
        history = self.model.fit_generator(
            self.data_loader.get_train_datagen(),
            epochs=1000,
            verbose=1,
            validation_data=self.data_loader.get_val_datagen(),
            callbacks=self.callbacks,
            # use_multiprocessing=True,
            # workers=6
        )

        self._log_exp_info(history)


    def _log_exp_info(self, history):
        # exp_name,model,img_dim,n_params,n_epochs,min_loss,max_acc,train_loss,
        # train_acc,val_loss,val_acc,remarks
        acc = history.history['acc']
        val_acc = history.history['val_acc']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        with open(EXPS_PATH, mode='a') as exps:
            exps_writer = csv.writer(exps, delimiter=',')
            exps_writer.writerow([
                self.config.exp.name,
                str(self.config.model.toDict()),
                self.config.data_loader.img_dim,
                self.model.count_params(),
                len(val_loss) - 20,
                min(val_loss),
                max(acc),
                loss[-21],
                acc[-21],
                val_loss[-21],
                val_acc[-21],
                '',
            ])
