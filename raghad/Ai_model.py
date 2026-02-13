import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# 1. Load Excel data
df = pd.read_excel("raghad/larger_dataset.xlsx") #assume this is th file 

# 2. Separate features and labels
features = df.drop(columns=['Word']).values  
labels = df['Word'].values

# 3. Encode labels to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 4. Convert labels to one-hot (required for classification)
labels_onehot = to_categorical(labels_encoded)
num_classes = len(label_encoder.classes_)

frames_per_word = 250 # we have 250 samples per word in our code 
num_words = len(features) // frames_per_word
X_sequences = features.reshape(num_words, frames_per_word, -1) 
y_sequences = labels_onehot[::frames_per_word]


# 5. Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_features = X_sequences.shape[2]
X_scaled = X_sequences.reshape(-1, num_features)
X_scaled = scaler.fit_transform(X_scaled)
X_scaled = X_scaled.reshape(num_words, frames_per_word, num_features)

# 6. Train-test split
from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y_sequences, test_size=0.2, random_state=42, shuffle=True
)

# 7. Split training into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, shuffle=True
)

# 8. Build unidirectional LSTM model
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.LSTM(256, activation='tanh', return_sequences=True, input_shape=(frames_per_word, num_features)),
    layers.Dropout(0.2),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64, activation='tanh'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # classification output
])

from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 9. Callbacks
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
early_stopper = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
model_check = ModelCheckpoint('best_gesture_model.keras', save_best_only=True, monitor='val_loss')

# 10. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # it was 50 i changed it 
    batch_size=4, # it was 64 but i changed it because of the small data set for now
    verbose=1,
    callbacks=[early_stopper, model_check, lr_scheduler]
)

# 11. Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

model.save('best_gesture_model.keras') 

