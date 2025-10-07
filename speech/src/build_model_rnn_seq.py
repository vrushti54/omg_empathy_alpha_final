# build_model_rnn_seq.py
# CNN + BiGRU sequence model (CPU-friendly) with proper 250-frame output

import os, time, configparser
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import (Input, Conv1D, MaxPooling1D, Bidirectional, GRU,
                          BatchNormalization, Dropout, TimeDistributed,
                          Dense, Flatten)
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             CSVLogger)
from keras import optimizers, regularizers

import utilities_func as uf
import loadconfig

np.random.seed(1)

print("loading dataset...")
cfg = configparser.ConfigParser()
cfg.read(loadconfig.load())

# ---- paths from config
SAVE_MODEL            = cfg.get('model', 'save_model')            # where to save best .h5
TRAIN_X_PATH          = cfg.get('model', 'training_predictors_load')
TRAIN_Y_PATH          = cfg.get('model', 'training_target_load')
VAL_X_PATH            = cfg.get('model', 'validation_predictors_load')
VAL_Y_PATH            = cfg.get('model', 'validation_target_load')
SEQ_LENGTH            = cfg.getint('preprocessing', 'sequence_length')  # should be 250

# ---- load datasets
X_tr = np.load(TRAIN_X_PATH).astype("float32")
y_tr = np.load(TRAIN_Y_PATH).astype("float32")
X_va = np.load(VAL_X_PATH).astype("float32")
y_va = np.load(VAL_Y_PATH).astype("float32")

# ---- normalization (per-feature if available)
feat_dir  = os.path.dirname(TRAIN_X_PATH)
mean_path = os.path.join(feat_dir, 'train_mean.npy')
std_path  = os.path.join(feat_dir, 'train_std.npy')

if os.path.exists(mean_path) and os.path.exists(std_path):
    mean = np.load(mean_path).astype('float32')        # (F,)
    std  = np.load(std_path ).astype('float32') + 1e-8
    print(f"[norm] per-feature stats loaded: {mean_path}, {std_path}")
else:
    mean = np.float32(X_tr.mean())
    std  = np.float32(X_tr.std() + 1e-8)
    print(f"[norm] fallback: scalar mean/std -> mean={mean:.6f} std={std:.6f}")

X_tr = (X_tr - mean) / std
X_va = (X_va - mean) / std

# ---- shapes
T, F = int(X_tr.shape[1]), int(X_tr.shape[2])
print(f"shapes  X_train={X_tr.shape}  y_train={y_tr.shape}")
print(f"        X_val  ={X_va.shape}  y_val  ={y_va.shape}")

assert T == SEQ_LENGTH, f"Config sequence_length={SEQ_LENGTH} but X has T={T}"

# ---- hyperparameters
batch_size = 16
num_epochs = 20
conv_filters = 96
kernel = 5
gru1_units = 128
gru2_units = 64
td_hidden  = 32
drop_prob  = 0.30
reg = regularizers.l2(1e-3)

opt = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True, clipnorm=1.0)

def batch_CCC(y_true, y_pred):
    # uf.CCC returns sum over batch; average it so "lower is better"
    c = uf.CCC(y_true, y_pred) / float(batch_size)
    return 1.0 - c

# ---- model
inp = Input(shape=(T, F))                            # (250, 79)

x = Conv1D(conv_filters, kernel, padding='same', activation='linear')(inp)
x = BatchNormalization()(x)
x = Dropout(drop_prob)(x)

x = Conv1D(conv_filters, kernel, padding='same', activation='linear')(x)
x = BatchNormalization()(x)

x = MaxPooling1D(pool_size=2)(x)                     # time: 250 -> 125

x = Bidirectional(GRU(gru1_units, return_sequences=True))(x)   # (125, 256)
x = Bidirectional(GRU(gru2_units, return_sequences=True))(x)   # (125, 128)
x = BatchNormalization()(x)

x = TimeDistributed(Dense(td_hidden, activation='linear', kernel_regularizer=reg))(x)  # (125, 32)
x = Dropout(drop_prob)(x)
x = TimeDistributed(Dense(1, activation='linear'))(x)                                   # (125, 1)

x = Flatten()(x)                          # -> (125,)
out = Dense(T, activation='linear')(x)    # project back to 250 to match targets

model = Model(inp, out)
model.compile(loss=batch_CCC, optimizer=opt)
print(model.summary())

# ---- logs / callbacks
os.makedirs(os.path.join("speech", "runs", "checkpoints"), exist_ok=True)
os.makedirs(os.path.join("speech", "runs", "logs"), exist_ok=True)
os.makedirs(os.path.join("speech", "runs", "plots"), exist_ok=True)

# Make a timestamped checkpoint path to avoid colliding with old models
ts = time.strftime("%Y%m%d_%H%M%S")
ckpt_path = SAVE_MODEL
base, ext = os.path.splitext(SAVE_MODEL)
ckpt_path = f"{base}_{ts}{ext}"

with open(os.path.join("speech", "runs", "logs", f"model_summary_{ts}.txt"), "w") as f:
    model.summary(print_fn=lambda s: f.write(s + "\n"))
with open(os.path.join("speech", "runs", "logs", f"model_arch_{ts}.json"), "w") as f:
    f.write(model.to_json())
print(f"[save] model summary -> speech/runs/logs/model_summary_{ts}.txt")
print(f"[save] model arch json -> speech/runs/logs/model_arch_{ts}.json")

ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True,
                       mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2,
                          min_lr=1e-6, verbose=1)
csvlog = CSVLogger(os.path.join("speech", "runs", "logs", f"train_log_{ts}.csv"))
print(f"[log]  csv logger -> speech/runs/logs/train_log_{ts}.csv")

# ---- train
hist = model.fit(
    X_tr, y_tr,
    epochs=num_epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(X_va, y_va),
    callbacks=[ckpt, es, rlrop, csvlog],
    verbose=1
)

print("Train loss =", min(hist.history['loss']))
print("Validation loss =", min(hist.history['val_loss']))

# ---- plot curves
plt.figure(figsize=(8,6))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.title('MODEL PERFORMANCE', size=16)
plt.ylabel('loss', size=14)
plt.xlabel('Epoch', size=14)
plt.legend(fontsize=12)
plt.tight_layout()

png = os.path.join("speech", "runs", "plots", f"training_loss_{ts}.png")
plt.savefig(png, dpi=150); plt.show(); plt.close()
print(f"[save] Training curve saved to {png}")
print(f"[ckpt] Best model saved to: {ckpt_path}")
