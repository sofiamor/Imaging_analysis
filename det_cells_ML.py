import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Load the base model, which could be something like VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Example for binary classification, adjust for your task

model = Model(inputs=base_model.input, outputs=output_layer)

# Freeze the base model layers so they are not trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('C:/Users/sofik/Desktop/traning', target_size=(256, 256), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory('C:/Users/sofik/Desktop/validation', target_size=(256, 256), batch_size=32, class_mode='binary')

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(train_generator, epochs=50, validation_data=val_generator, callbacks=callbacks)

# Evaluate the model
model.evaluate(val_generator)
