
import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from LSTM import LoadData, ClassMapper, SessionSplit, GestureSeq, Lstm


# ------------------------------------------------------------------ #

#i will chose only the first 95% of the data 3shan the last data will probably have unimportant values ( me finishing the word and going to click enter)
def PickTargetT(train_df: pd.DataFrame, percentile: float = 95.0) -> int:
    return int(np.ceil(np.percentile(train_df["n_timesteps"], percentile)))


def DefaultValSessions(df: pd.DataFrame) -> list:
    sessions = sorted(df["session"].unique())
    if len(sessions) < 2:
        raise ValueError(
            "Need at least 2 sessions for a session-based split. "
            f"Manifest only has: {sessions}"
        )
    return [sessions[-1]]


# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                        help="Path to Logs.csv produced by collect.py.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save outputs. Defaults to runs/<timestamp>.")
    parser.add_argument("--val-sessions", nargs="+", default=None,
                        help="Session ids to use for validation. Default: latest session.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--lstm1",      type=int, default=128)
    parser.add_argument("--lstm2",      type=int, default=64)
    parser.add_argument("--dense",      type=int, default=64)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--l2",         type=float, default=1e-4)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training-time augmentation (useful for debugging).")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Output directory
    if args.out_dir is None:
        args.out_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load manifest and build label map -------------------------
    df = LoadData(args.manifest)
    word_to_id, id_to_word = ClassMapper(df)
    num_classes = len(word_to_id)
    print(f"Loaded {len(df)} samples | {num_classes} classes "
          f"| {df['session'].nunique()} sessions")

    # ---- Session split ---------------------------------------------
    val_sessions = args.val_sessions or DefaultValSessions(df)
    print(f"Validation sessions: {val_sessions}")
    train_df, val_df = SessionSplit(df, val_sessions)
    print(f"Train: {len(train_df)} samples   Val: {len(val_df)} samples")

    if len(val_df) == 0:
        raise ValueError("Validation split is empty. Check --val-sessions.")

    # ---- Padding length (decided from training data only) ----------
    target_T = PickTargetT(train_df, percentile=95.0)
    print(f"Padding length T = {target_T}  (95th percentile of train n_timesteps)")

    # ---- Feature count (read from the first file) ------------------
    sample_arr = np.load(train_df.iloc[0]["filepath"])
    num_features = sample_arr.shape[1]
    print(f"Feature count F = {num_features}")

    # ---- Data sequences --------------------------------------------
    train_seq = GestureSeq(
        df=train_df,
        word_to_id=word_to_id,
        TargetT=target_T,
        batch_size=args.batch_size,
        augment=not args.no_augment,
        shuffle=True,
        seed=args.seed,
        num_classes=num_classes,
    )
    val_seq = GestureSeq(
        df=val_df,
        word_to_id=word_to_id,
        TargetT=target_T,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        seed=args.seed,
        num_classes=num_classes,
    )

    # ---- Build and compile model -----------------------------------
    model = Lstm(
        featuresNum=num_features,
        classesNum=num_classes,
        TimeSteps=target_T,
        units1=args.lstm1,
        units2=args.lstm2,
        dense=args.dense,
        l2=args.l2,
        DROPOUT=args.dropout,
    )
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ----------------------------------------------------
    best_path = os.path.join(args.out_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        CSVLogger(os.path.join(args.out_dir, "history.csv")),
    ]
    

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({
            **vars(args),
            "target_T":    target_T,
            "num_features": num_features,
            "num_classes":  num_classes,
            "val_sessions": val_sessions,
        }, f, indent=2)

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({
            "word_to_id": word_to_id,
            "id_to_word": {str(k): v for k, v in id_to_word.items()},
        }, f, indent=2, ensure_ascii=False)

    # ---- Train -----------------------------------------------------
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nDone. Best model: {best_path}")
    print(f"History:          {os.path.join(args.out_dir, 'history.csv')}")


if __name__ == "__main__":
    main()