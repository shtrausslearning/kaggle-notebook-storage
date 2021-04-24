# PRETRAINED ARCHITECTURE MODELS

def pretrained_model(head_id):

    # Define model with different applications
    model = Sequential()

    ''' Define Head Pretrained Models '''

    if(head_id is 'vgg'):
        model.add(app.VGG16(input_shape=sshape,
                            pooling='avg',
                            classes=1000,
                            include_top=False,
                            weights='imagenet'))

    elif(head_id is 'resnet'):
        model.add(app.ResNet101(include_top=False,
                               input_tensor=None,
                               input_shape=sshape,
                               pooling='avg',
                               classes=100,
                               weights='imagenet'))

    elif(head_id is 'mobilenet'):
        model.add(app.MobileNet(alpha=1.0,
                               depth_multiplier=1,
                               dropout=0.001,
                               include_top=False,
                               weights="imagenet",
                               input_tensor=None,
                               input_shape = sshape,
                               pooling=None,
                               classes=1000))

    elif(head_id is 'inception'):
        model.add(InceptionV3(input_shape = sshape, 
                                 include_top = False, 
                                 weights = 'imagenet'))

    elif(head_id is 'efficientnet'):
        model.add(EfficientNetB4(input_shape = sshape, 
                                    include_top = False, 
                                    weights = 'imagenet'))

    ''' Tail Model Part '''
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(labels,activation='softmax'))

    # # freeze main model coefficients
    model.layers[0].trainable = False

    return model
  
# IMPLEMENTATATION
  
''' Define Model Architectre '''
lst_heads = ['vgg','resnet','mobilenet','inception','efficientnet']
# lst_heads = ['mobilenet']
verbose = True

lst_res_train = [] ; lst_res_val = []
for head_id in lst_heads:

    # define CNN head model
    model = pretrained_model(head_id)

    ''' Callback Options During Training '''
    callbacks = [ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=0, 
                                   factor=0.5,mode='max',min_lr=0.001),
                 ModelCheckpoint(filepath=f'model_{head_id}.h5',monitor='val_accuracy',
                                 mode = 'max',verbose=0,save_best_only=True),
                 TqdmCallback(verbose=1)] 

    ''' Model Compilation '''
    # Let's use Adam optimiser,categorical_crossentropy for the loss function & accuracy for the evaluation metric. 
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

    ''' Start Training '''
    start = time.time()
    history = model.fit(gen_train,
                        validation_data = gen_valid,
                        callbacks=callbacks,
                        verbose=0,epochs=n_epochs)
    end = time.time()
    if(verbose):
        print(f'Head Model: {head_id}')
        print(f'The time taken to execute is {round(end-start,2)} seconds.')
        print(f'Maximum Train/Val {max(history.history["accuracy"]):.4f}/{max(history.history["val_accuracy"]):.4f}')
        
    # store results 
    lst_res_train.append(history.history['accuracy'])  
    lst_res_val.append(history.history['val_accuracy']) 


