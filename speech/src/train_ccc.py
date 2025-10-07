# -*- coding: utf-8 -*-
import os, time, argparse, configparser
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, BatchNormalization, MaxPooling1D
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, Flatten
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau

# -----------------------
# CCC in TensorFlow
# -----------------------
def tf_ccc(y_true, y_pred, eps=1e-8):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mu_t = tf.reduce_mean(y_true, axis=[-1], keepdims=True)
    mu_p = tf.reduce_mean(y_pred, axis=[-1], keepdims=True)
    v_t  = tf.reduce_mean((y_true - mu_t)**2, axis=[-1])
    v_p  = tf.reduce_mean((y_pred - mu_p)**2, axis=[-1])
    cov  = tf.reduce_mean((y_true - mu_t)*(y_pred - mu_p), axis=[-1])
    ccc  = (2.0*cov) / (v_t + v_p + (mu_t[...,0]-mu_p[...,0])**2 + eps)
    return ccc  # shape (batch,)

@tf.function
def neg_ccc_metric(y_true, y_pred):
    return -tf.reduce_mean(tf_ccc(y_true, y_pred))

@tf.function
def mean_ccc_metric(y_true, y_pred):
    return tf.reduce_mean(tf_ccc(y_true, y_pred))

def ccc_loss(alpha_ccc):
    # blended: (1-alpha)*MSE + alpha*(1-CCC)
    def _loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[-1])
        ccc = tf_ccc(y_true, y_pred)
        return (1.0 - alpha_ccc)*mse + alpha_ccc*(1.0 - ccc)
    return _loss

# -----------------------
# Model
# -----------------------
def make_model(seq_len, feat_dim):
    inp = Input(shape=(seq_len, feat_dim))
    x = Conv1D(96, 5, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x); x = Dropout(0.1)(x)
    x = Conv1D(96, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64,  return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(32, activation='relu'))(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(1))(x)
    x = Flatten()(x)
    out = Dense(seq_len)(x)
    return Model(inp, out)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    # data
    TRAIN_X = cfg.get('model', 'training_predictors_load', fallback="speech/runs/features/train_X.npy")
    TRAIN_Y = cfg.get('model', 'training_target_load',    fallback="speech/runs/features/train_y.npy")
    VAL_X   = cfg.get('model', 'validation_predictors_load', fallback="speech/runs/features/val_X.npy")
    VAL_Y   = cfg.get('model', 'validation_target_load',    fallback="speech/runs/features/val_y.npy")
    SEQ_LEN = cfg.getint('preprocessing', 'sequence_length', fallback=250)

    X_tr = np.load(TRAIN_X).astype(np.float32)
    y_tr = np.load(TRAIN_Y).astype(np.float32)
    X_va = np.load(VAL_X).astype(np.float32)
    y_va = np.load(VAL_Y).astype(np.float32)

    feat_dim = X_tr.shape[-1]

    # hparams
    batch_size   = cfg.getint('training', 'batch_size', fallback=64)
    epochs       = cfg.getint('training', 'epochs', fallback=40)
    lr           = cfg.getfloat('training', 'learning_rate', fallback=5e-4)
    alpha_ccc    = cfg.getfloat('training', 'alpha_ccc', fallback=0.8)

    # model
    model = make_model(SEQ_LEN, feat_dim)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss=ccc_loss(alpha_ccc),
                  metrics=[mean_ccc_metric, neg_ccc_metric])  # mean_ccc_metric reported as 'mean_ccc_metric'

    print(model.summary())

    # checkpoints by **val mean CCC (maximize)**
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = "speech/runs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"best_model_ccc_{ts}.h5")

    mcp = ModelCheckpoint(
        ckpt_path,
        monitor="val_mean_ccc_metric",  # maximize CCC
        mode="max",
        save_best_only=True,
        save_weights_only=False
    )
    csv = CSVLogger(f"speech/runs/logs/train_ccc_{ts}.csv")
    es  = EarlyStopping(monitor="val_mean_ccc_metric", mode="max", patience=5, restore_best_weights=True)
    rl  = ReduceLROnPlateau(monitor="val_mean_ccc_metric", mode="max", factor=0.6, patience=2, verbose=1)

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[mcp, csv, es, rl],
        verbose=1
    )

    # convenience symlink/copy
    import shutil
    shutil.copy2(ckpt_path, os.path.join(ckpt_dir, "best_model.h5"))
    print(f"[ckpt] Best model (by val CCC) -> {ckpt_path}")

if __name__ == "__main__":
    main()
