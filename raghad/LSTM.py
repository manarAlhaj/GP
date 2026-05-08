"""
Pipeline:
    Training:
        load raw .npy  (T, 11)
        → preprocess_single()        → (100, 16)
        → AugmentSample()            → (100, 16)  training only
        → apply_zscore()             → (100, 16)
        → LSTM

    Validation:
        load raw .npy  (T, 11)
        → preprocess_single()        → (100, 16)
        → apply_zscore()             → (100, 16)
        → LSTM
"""
from keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np
from keras import layers, models, regularizers
from preprocessing import preprocess_single, AugmentSample, apply_zscore, compute_signer_stats


# ══════════════════════════════════════════════════════════════
# LOADING + SPLITTING
# ══════════════════════════════════════════════════════════════

#loading data into data frames
def LoadData(LogPath: str) -> pd.DataFrame:
    df = pd.read_csv(LogPath)
    required = {"filepath", "word", "session", "n_timesteps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Log File missing required columns: {missing}")
    return df

def ClassMapper(df: pd.DataFrame):
    words = sorted(df["word"].unique())
    word_to_id = {w: i for i, w in enumerate(words)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    return word_to_id, id_to_word

def SessionSplit(df: pd.DataFrame, ValSess):
    validition = df['session'].isin(ValSess)
    Train = df[~validition].reset_index(drop=True)
    Val = df[validition].reset_index(drop=True)
    return Train, Val

# ══════════════════════════════════════════════════════════════
# COMPUTE NORMALIZATION STATS — call once on training data only
# ══════════════════════════════════════════════════════════════
def compute_global_stats(df, target_length=100):
    """
    Preprocesses every training recording and computes mean + std.
    Call this ONCE on Train_df only — never on Val_df.

    Returns: mean (16,), std (16,)
    """
    all_processed = []
    print("Computing normalization stats from training data...")

    for _, row in df.iterrows():
        arr       = np.load(row["filepath"]).astype(np.float32)  # (T, 11)
        processed = preprocess_single(arr, target_length)         # (100, 16)
        all_processed.append(processed)

    mean, std = compute_signer_stats(all_processed)
    print(f"Stats computed from {len(all_processed)} recordings.")
    return mean, std


# ══════════════════════════════════════════════════════════════
# DATASET SEQUENCE
# ══════════════════════════════════════════════════════════════

class GestureSeq(Sequence):
    def __init__(self,
            df: pd.DataFrame,
            word_to_id: dict,
            global_mean: np.ndarray,       # (16,) — from training split only add this
            global_std:  np.ndarray,        # (16,) — from training split only add this
            TargetT: int = 100,
            batch_size: int = 32,
            augment: bool = False,
            shuffle: bool = True,
            seed: int = 42,
            num_classes: int = None
    ):
        self.df = df.reset_index(drop=True)
        self.word_to_id = word_to_id
        self.num_classes = num_classes or len(word_to_id)
        self.TargetT = TargetT
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.mean = global_mean  ## add this 
        self.std = global_std           ## add this 
        self.rng = np.random.default_rng(seed)
        self._indices = np.arange(len(self.df))

        if self.shuffle:
            self.rng.shuffle(self._indices)
        
    def __len__(self): #num of batches per epoch
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self): ## the keras name ? 
        if self.shuffle:
            self.rng.shuffle(self._indices)
            
    def __getitem__(self, idx):
        batch_idx = self._indices[idx * self.batch_size : (idx + 1) * self.batch_size]  
        
        x = np.zeros((len(batch_idx), self.TargetT, 16), dtype=np.float32)  # ✅ fix this 
        y = np.zeros((len(batch_idx),), dtype=np.int64) #hold classes 
        
        for i, row_i in enumerate(batch_idx):
            row = self.df.iloc[row_i]
            
            # 1 — load raw recording
            arr = np.load(row["filepath"]).astype(np.float32)     # (T, 11)

            # 2 — preprocess first                                 # ✅ fix 1, 2, 3
            arr = preprocess_single(arr, self.TargetT)
            
            # 3 — augment AFTER preprocessing, training only       # ✅ fix 1
            if self.augment:
                arr = AugmentSample(arr, rng=self.rng)             # (100, 16)

            # 4 — z-score normalize                                # ✅ fix 4
            arr = apply_zscore(arr, self.mean, self.std)           # (100, 16)

            x[i] = arr
            y[i] = self.word_to_id[row["word"]]
        return x, to_categorical(y, num_classes=self.num_classes)



# do we need this ? 
    def _infer_F(self):
       #sneak peak on data bs, notice its called in getitem as its part of detecting num of feature.
        if not hasattr(self, "_F_cached"):
            first = np.load(self.df.iloc[0]["filepath"])
            self._F_cached = first.shape[1]
        return self._F_cached

# ══════════════════════════════════════════════════════════════
# LSTM MODEL
# ══════════════════════════════════════════════════════════════

def Lstm ( 
         featuresNum : int,
         classesNum : int,
         TimeSteps: int,
         units1: int = 128,
         units2: int = 64,
         dense: int = 64,
         l2 : float = 1e-3,
         DROPOUT : float = 0.3) -> models.Model:
    
# ✅ fix 7 — Masking removed: every timestep is real after resampling

    INPUT = layers.Input(shape=(TimeSteps, featuresNum), name = "SensorInput") 

#mask = layers.Masking(mask_value = maskval, name = "Masking layer")(INPUT)

    LS1 = layers.LSTM(units=units1, return_sequences=True, kernel_regularizer=regularizers.l2(l2), 
                      recurrent_regularizer=regularizers.l2(l2), name = "LSTM_1")(INPUT)
    
    DROP1 = layers.Dropout(DROPOUT, name = "Dropout_1")(LS1)

    LS2 = layers.LSTM(units=units2, return_sequences=False, kernel_regularizer=regularizers.l2(l2), 
                      recurrent_regularizer=regularizers.l2(l2), name = "LSTM_2")(DROP1)
    
    DROP2 = layers.Dropout(DROPOUT, name = "Dropout_2")(LS2)

    DENSE1 = layers.Dense(dense, activation="relu", kernel_regularizer=regularizers.l2(l2), name = "Dense_1")(DROP2)

    DROP3 = layers.Dropout(DROPOUT, name = "Dropout_3")(DENSE1)

    OUTPUT = layers.Dense(classesNum, activation="softmax", name = "Output_SoftMax")(DROP3)

    MODEL = models.Model(inputs=INPUT, outputs=OUTPUT, name="LSTM Gestture Recognition Model")

    return MODEL


if __name__ == "__main__":
    mod = Lstm(featuresNum=16, classesNum=11, TimeSteps=100)
    mod.summary()
    

    

    

    
