
"""
for training:
    load raw .npy        (gathered by us)         
    apply random noise %5             
    apply random per sensor scale 30%
    apply random time stretch/compress (gesture duration) 10%     
    feed to model

for validation:
    load raw .npy                 
    feed to model
"""
from keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np
from sklearn import preprocessing as sk 
import os
import csv 
from scipy.interpolate import interp1d
import tensorflow as tf
from keras import layers, models, regularizers



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

# *********************************************************************** # 
# Data Augmentation 

def AugmentSample(arr: np.ndarray,
                NoiseFrac: float = 0.05,
                ScaleRange: float = 0.10,
                TimeRange: float = 0.30,
                rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)
    arr = arr.astype(np.float32, copy=True)
    T, F = arr.shape
#noise
    PerSensorStd = arr.std(axis=0, keepdims=True)
    noise = rng.standard_normal(size = arr.shape) * PerSensorStd * NoiseFrac
    arr += noise.astype(np.float32)
    #scaling
    scale = rng.uniform(1.0  - ScaleRange, 1.0  + ScaleRange, size=(1, F)).astype(np.float32)
    arr *= scale
    #time warping + interpolation
    timewarp  = rng.uniform(1.0 - TimeRange, 1.0  + TimeRange) 
    NewT = max(4, int(round(T * timewarp)))
    if NewT != T:
        old_x = np.arange(T, dtype=np.float32)
        new_x = np.linspace(0, T - 1, NewT, dtype=np.float32)
        f = interp1d(old_x, arr, axis=0, kind="linear")
        arr = f(new_x).astype(np.float32)
    return arr

# *********************************************************************** # 
# zero padding 
def ZeroPad(arr: np.ndarray, TargetT: int) -> np.ndarray:
    T, F = arr.shape
    if T >= TargetT:
        return arr[:TargetT]
    Padding = np.zeros((TargetT - T, F), dtype=np.float32)
    return np.concatenate([arr, Padding], axis=0)

# *********************************************************************** # 

class GestureSeq(Sequence):
    def __init__(self,
            df: pd.DataFrame,
            word_to_id: dict,
            TargetT: int,
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
        self.rng = np.random.default_rng(seed)
        self._indices = np.arange(len(self.df))

        if self.shuffle:
            self.rng.shuffle(self._indices)
        
    def __len__(self): #num of batches per epoch
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def EpochEnd(self):
        if self.shuffle:
            self.rng.shuffle(self._indices)
            
    def __getitem__(self, idx):
        batch_idx = self._indices[idx * self.batch_size : (idx + 1) * self.batch_size]  
        x = np.zeros((len(batch_idx), self.TargetT, self._infer_F()), dtype=np.float32) # input gesture seq
        y = np.zeros((len(batch_idx),), dtype=np.int64) #hold classes 
        for i, row_i in enumerate(batch_idx):
            row = self.df.iloc[row_i]
            arr = np.load(row["filepath"]).astype(np.float32)
            if self.augment:
                arr = AugmentSample(arr, rng=self.rng)
            arr = ZeroPad(arr, self.TargetT)
            x[i] = arr
            y[i] = self.word_to_id[row["word"]]
        return x, to_categorical(y, num_classes=self.num_classes)

    def _infer_F(self):
       #sneak peak on data bs, notice its called in getitem as its part of detecting num of feature.
        if not hasattr(self, "_F_cached"):
            first = np.load(self.df.iloc[0]["filepath"])
            self._F_cached = first.shape[1]
        return self._F_cached

# *********************************************************************** #

def Lstm ( 
         featuresNum : int,
         classesNum : int,
         TimeSteps: int,
         units1: int = 128,
         units2: int = 64,
         dense: int = 64,
         l2 : float = 1e-3,
         DROPOUT : float = 0.3,
         maskval : float = 0.0,) -> models.Model:
    
    INPUT = layers.Input(shape=(TimeSteps, featuresNum), name = "SensorInput") 

    mask = layers.Masking(mask_value = maskval, name = "Masking layer")(INPUT)

    LS1 = layers.LSTM(units=units1, return_sequences=True, kernel_regularizer=regularizers.l2(l2), 
                      recurrent_regularizer=regularizers.l2(l2), name = "LSTM_1")(mask)
    
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
    mod = Lstm(featuresNum=11, classesNum=11, TimeSteps=120)
    mod.summary()
    

    

    

    
