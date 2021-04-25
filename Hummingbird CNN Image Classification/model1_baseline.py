# Example Keras Preload Images from Training/Validation & Test Folders using ImageDataGenerator

# \_ Train 
#  |_ Class 1
#  |_ Class 2
#  |_ ...

# \_ Validation 
#  |_ Class 1
#  |_ Class 2
#  |_ ...
# ...

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Define DataGenerators w/ default NN augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255)
gen_datagen = ImageDataGenerator(rescale=1.0/255)

# DataGenerators via Folder Directory
gen_train = train_datagen.flow_from_directory(train_folder, 
                        target_size=(224,224),  # target size
                        batch_size=32,          # batch size
                        class_mode='categorical')    # batch size

gen_valid = gen_datagen.flow_from_directory(val_folder,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical')

gen_test = gen_datagen.flow_from_directory(test_folder,
                        target_size=(224,224),
                        batch_size=32,
                        class_mode='categorical')

# Define Simple Convolution Neural Network 

# Two Convolution Layer CNN (
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=(224,224)),    
    keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(labels, activation="softmax")
])

# show CNN architecture & all available weight numbers
model.summary()

# ''' Model Compilation '''
model.compile(optimizer='Adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ''' Callback Options During Training '''
callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                               factor=0.5,mode='max',min_lr=0.001),
             ModelCheckpoint(filepath=f'model_cnn.h5',monitor='val_accuracy',
                             mode = 'max',verbose=0,save_best_only=True),
             TqdmCallback(verbose=1)] 

''' Start Training '''
start = time.time()
history = model.fit(gen_train,
                    validation_data = gen_valid,
                    callbacks=callbacks,
                    verbose=0,epochs=n_epochs)
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')
print(f'Maximum Train/Val {max(history.history["accuracy"]):.4f}/{max(history.history["val_accuracy"]):.4f}')
