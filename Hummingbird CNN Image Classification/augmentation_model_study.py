# Evaluate CNN model w/ imported augmentation, input -> list of augmentations.
# Sample usage; https://www.kaggle.com/shtrausslearning/hummingbird-classification-with-cnn

def augment_model(lst_aug):
    
    # Requires list of augmentations to training & val/test data

    # Define DataGenerators
    gen_train = lst_aug[0].flow_from_directory(train_folder, 
                            target_size=(224,224),  # target size
                            batch_size=32,          # batch size
                            class_mode='categorical')    # batch size

    gen_valid = lst_aug[1].flow_from_directory(val_folder,
                            target_size=(224,224),
                            batch_size=32,
                            class_mode='categorical')

    gen_test = lst_aug[1].flow_from_directory(test_folder,
                            target_size=(224,224),
                            batch_size=32,
                            class_mode='categorical')

    # Define a CNN Model (
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

    # Callback Options During Training 
    callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                                   factor=0.5,mode='max',min_lr=0.001),
                 ModelCheckpoint(filepath=f'model_resnet34.h5',monitor='val_accuracy',
                                 mode = 'max',verbose=0,save_best_only=True),
                 TqdmCallback(verbose=1)] 
    
    # Compile Model
    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    # Evaluate Model
    history = model.fit(gen_train,
                        validation_data = gen_valid,
                        callbacks=callbacks,
                        verbose=0,epochs=n_epochs)
    
    # Return Result History
    return history 

  
# List of Augmentation Options & default settings 
lst_augopt = ['rescale','horizontal_flip','vertical_flip',
              'brightness_range','rotation_range','shear_range',
              'zoom_range','width_shift_range','height_shift_range',
              'channel_shift_range','zca_whitening','featurewise_center',
              'samplewise_center','featurewise_std_normalization','samplewise_std_normalization']
lst_augval = [1.0/255,True,True,  
              [1.1,1.5],0.2,0.2,
              0.2,0,0,
              0,True,False,
             False,False,False]

# Select Augmentations 
lst_select = [[0,1],[0,2],[0,3],[0,1,5,6]]   # augmentation selection
ii=-1; lst_res_train = [] ; lst_res_val = []
for augs in lst_select:
    
    print('Augmentation Combination')
    # get dictionary of augmentation options
    ii+=1; dic_select = dict(zip([lst_augopt[i] for i in lst_select[ii]],[lst_augval[i] for i in lst_select[ii]]))
    print(dic_select)

    # define augmentation options
    train_datagen = ImageDataGenerator(**dic_select) # pass arguments
    gen_datagen = ImageDataGenerator(rescale=1.0/255)

    # evaluate model & return history metric
    history = augment_model([train_datagen,gen_datagen])

    # store results
    lst_res_train.append(history.history['accuracy'])  
    lst_res_val.append(history.history['val_accuracy'])
    
# PLOT RESULTS 

fig = make_subplots(rows=1, cols=2,subplot_titles=['Training Accuracy','Validation Accuracy'])
ii=-1
for i in lst_select:
    ii+=1;fig.add_trace(go.Scatter(x=[i for i in range(0,n_epochs)], 
                                   y=lst_res_train[ii],
                                   name=f'{i}',
                                   line=dict(color=lst_color[ii])),row=1, col=1)
ii=-1
for i in lst_select:
    ii+=1;fig.add_trace(go.Scatter(x=[i for i in range(0,n_epochs)], 
                                   y=lst_res_val[ii],
                                   name=f'{i}',
                                   line=dict(color=lst_color[ii])),row=1, col=2)
    
fig.update_layout(template='plotly_white',
                  title='<b> Image Augmentation Variations : Evaluation Metric (Accuracy)</b>',
                 font=dict(family='sans-serif',size=14))
fig.update_layout(margin={"r":0,"t":100,"l":0,"b":0},height=400,showlegend=False)
fig.show()

