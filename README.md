HAND GESTURE IMAGE DATASET TO TEXT CONVERSION..

# STEP-BY-STEP PROCEDURE..

# 1. IMPORTING THE LIBRARIES..
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.optimizers import Adam

# 2. DEFINE PATHS..
 train_data_dir = r"C:\Users\jocel\OneDrive\Desktop\joseline vincy\asl_alphabet_test\train"
 test_data_dir = r"C:\Users\jocel\OneDrive\Desktop\joseline vincy\asl_alphabet_test\test"

# 3.SET IMAGE PARAMETERS..
img_width, img_height = 64, 64
batch_size = 32

# 4.DATA AUGUMENTATION AND PRE-PROCESSING..
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 5.CREATE TRAIN GENERATOR..
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
) 
# A SUMMARY OF TRAINING DATA BE LIKE..
Found X images belonging to Y classes.

# 6.CREATE A TEST DATASET..
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
) 
# A SUMMARY OF TRAINING DATA BE LIKE..
Found A images belonging to B classes.

# 7.BUILD THE MODEL..
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 8.COMPILE THE MODEL..
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 9.TRAIN THE MODEL..
model.fit(train_generator, epochs=1, validation_data=test_generator)
# OUTPUT BE LIKE FOR THE TRAINING PROGRESS AND EPOCHS..
(Epoch 1/10
X batches
...
accuracy: 0.XX, val_accuracy: 0.YY

# 10.SAVE THE MODEL..
model.save('my_cnn_model.h5')

# 11.EVALUATE THE MODEL..
loss, accuracy = model.evaluate(test_generator)
# THE EVALUATION RESULTS
Test Accuracy: ZZ.ZZ%


























