from KERAS_MODELS import  *
from test_batchwise_training import generate_train_new,generate_val_new,generate_pred_new
from PARAMS import loadfeatureParams,createBatchTrainingParams
import sys
from keras.models import load_model
import datetime

featureParams = loadfeatureParams('featureParams.yaml')

# def train_and_predict(xTrain,yTrain,xVal,yVal,modelname,event,modelParams, predictOnData = None):


#     model = DNN(modelParams,modelname,event)




#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


#     #fit(self, x, y, batch_size=32, epochs=10, verbose=1, 
#     #callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
#     # class_weight=None, sample_weight=None, initial_epoch=0)

#     batch_size=modelParams[modelname][event]['batch_size']
#     epochs= modelParams[modelname][event]['epochs']
    
#     validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']
#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

#     hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
#                  validation_data = validation_data, 
#                  callbacks = [earlystopping], class_weight = class_weight)

#     print(hist.history)

#     if predictOnData is not None:
#         predictions = model.predict(predictOnData)

#     else:
#         predictions = model.predict(xVal)


#     return predictions,hist.history






# def train_and_predict_dnn(modelname,event,modelParams, directory,
#                       numTrainFiles = float('inf'), numValidfiles = float('inf')):

#     #createParamsFile()
#     modelParams = loadfeatureParams('modelParams.yaml')
#     model = DNN(modelParams,modelname,event)



#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


    
#     #batch_size=modelParams[modelname][event]['batch_size']
#     #epochs= modelParams[modelname][event]['epochs']
    
#     #validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']
#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')

#     batch_size=modelParams[modelname][event]['batch_size']
#     epochs = modelParams[modelname][event]['epochs']

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

#     #print(class_weight)
#     #sys.exit()

#     trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

#     trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

#     valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

#     valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

#     if modelParams['mycomp']:
#         trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
#         trainLabelFile = os.path.join("../../../../../", trainLabelFile)
#         valFeatureFile = os.path.join("../../../../../", valFeatureFile)
#         valLabelFile = os.path.join("../../../../../", valLabelFile)
#     else:
#         trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
#         trainLabelFile = os.path.join("/hdd4", trainLabelFile)
#         valFeatureFile = os.path.join("/hdd4", valFeatureFile)
#         valLabelFile = os.path.join("/hdd4", valLabelFile)


    
    
#     createBatchTrainingParams()
#     batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
    
#     #train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
#     #val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']

#     # for e in range(nb_epoch):
#     # print("epoch %d" % e)
#     # for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
#     #     model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)

#     train_total = batchTrainingParams[modelname]['generate_train_total']



#     for e in range(epochs):
        
#         for subEpochNumber in range(train_total):
#             print("epoch %d / %d" %(e,epochs-1))
#             print("subepoch %d / %d" %(subEpochNumber,train_total-1))
#             xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
#                                                 trainFeatureFile,
#                                                  trainLabelFile, numofFiles = float('inf'))
#             xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
#                                             valFeatureFile,
#                                          valLabelFile, numofFiles = float('inf'))



#             hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
#                                  verbose = 1, 
#                             validation_data = (xVal,yVal), callbacks = [earlystopping],
#                              class_weight = class_weight)





#     # hist = model.fit_generator(generate_train('dnn', featureParams, event, trainFeatureFile,
#     #                                  trainLabelFile
#     #                              ),
#     #                            steps_per_epoch=train_steps,
#     #                            epochs=epochs,verbose=1,
#     #                            callbacks = [earlystopping],
#     #                            validation_data=generate_val('dnn', featureParams, event, valFeatureFile,
#     #                                                     valLabelFile
#     #                                                 ),
#     #                            validation_steps=val_steps,
#     #                            class_weight=None
#     #                             )

#     # temp = model.layers[0]
#     # print(temp.get_weights())

#     # hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
#     #              validation_data = validation_data, callbacks = [earlystopping], class_weight = class_weight)

#     print(hist.history)

#     print("Done Training!")
#     print("Fetching Predictions ....")


#     # predictions = model.predict_generator( generate_pred('dnn', featureParams, event, valFeatureFile),
#     #                                             steps=1, verbose=1)
    
#     predictions = np.empty((0,1))
#     iterations = batchTrainingParams[modelname]['predict_iterations']
#     for i in range(iterations):
#         dataToPredictOn = generate_pred_new(i,modelname, featureParams,
#                                      event, valFeatureFile)
#         #print(model.predict(dataToPredictOn).shape)
#         temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

#         predictions = np.vstack((predictions, temp )).reshape((-1,1))        


#     print("final predictions.shape", predictions.shape)

#     return predictions,hist.history



# def train_and_predict_cnn(modelname,event,modelParams, directory,
#                       numTrainFiles = float('inf'), numValidfiles = float('inf')):

#     #createParamsFile()
#     modelParams = loadfeatureParams('modelParams.yaml')
#     model = CNN(modelParams,modelname,event)



#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


    
#     #batch_size=modelParams[modelname][event]['batch_size']
#     #epochs= modelParams[modelname][event]['epochs']
    
#     #validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']


#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')
#     earlystopping = None

#     batch_size=modelParams[modelname][event]['batch_size']
#     epochs = modelParams[modelname][event]['epochs']

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

#     #print(class_weight)
#     #sys.exit()

#     trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

#     trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

#     valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

#     valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

#     if modelParams['mycomp']:
#         trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
#         trainLabelFile = os.path.join("../../../../../", trainLabelFile)
#         valFeatureFile = os.path.join("../../../../../", valFeatureFile)
#         valLabelFile = os.path.join("../../../../../", valLabelFile)
#     else:
#         trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
#         trainLabelFile = os.path.join("/hdd4", trainLabelFile)
#         valFeatureFile = os.path.join("/hdd4", valFeatureFile)
#         valLabelFile = os.path.join("/hdd4", valLabelFile)

    
#     createBatchTrainingParams()
#     batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
    


#     train_total = batchTrainingParams[modelname]['generate_train_total']


#     verbose = 0;
#     for e in range(epochs):
        
#         for subEpochNumber in range(train_total):
#             verbose = 0

            
#             if(subEpochNumber%5 == 0):
#                 print("epoch %d / %d" %(e,epochs-1))
#                 print("subepoch %d / %d" %(subEpochNumber,train_total-1))
#                 verbose = 1
            
#             xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
#                                                 trainFeatureFile,
#                                                  trainLabelFile, numofFiles = float('inf'))
#             xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
#                                             valFeatureFile,
#                                          valLabelFile, numofFiles = float('inf'))



#             hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
#                                 verbose = verbose, 
#                             validation_data = (xVal,yVal), callbacks = None,
#                              class_weight = class_weight)




#     print(hist.history)

#     print("Done Training!")
#     print("Fetching Predictions ....")



    
#     predictions = np.empty((0,1))
#     iterations = batchTrainingParams[modelname]['predict_iterations']
#     for i in range(iterations):
#         dataToPredictOn = generate_pred_new(i,modelname, featureParams,
#                                      event, valFeatureFile)
#         #print(model.predict(dataToPredictOn).shape)
#         temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

#         predictions = np.vstack((predictions, temp )).reshape((-1,1))        


#     print("final predictions.shape", predictions.shape)

#     return predictions,hist.history



# def train_and_predict_rnn(modelname,event,modelParams, directory,
#                       numTrainFiles = float('inf'), numValidfiles = float('inf')):

#      #createParamsFile()
#     modelParams = loadfeatureParams('modelParams.yaml')
#     model = RNN(modelParams,modelname,event)



#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


    
#     #batch_size=modelParams[modelname][event]['batch_size']
#     #epochs= modelParams[modelname][event]['epochs']
    
#     #validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']


#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')
#     earlystopping = None

#     batch_size=modelParams[modelname][event]['batch_size']
#     epochs = modelParams[modelname][event]['epochs']

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

#     #print(class_weight)
#     #sys.exit()

#     trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

#     trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

#     valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

#     valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

#     newTestFeatureFile = directory+ '_' + event + '_' + 'newTest' + '_features.h5'

#     if modelParams['mycomp']:
#         trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
#         trainLabelFile = os.path.join("../../../../../", trainLabelFile)
#         valFeatureFile = os.path.join("../../../../../", valFeatureFile)
#         valLabelFile = os.path.join("../../../../../", valLabelFile)
#         newTestFeatureFile = os.path.join("../../../../../", newTestFeatureFile)
#     else:
#         trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
#         trainLabelFile = os.path.join("/hdd4", trainLabelFile)
#         valFeatureFile = os.path.join("/hdd4", valFeatureFile)
#         valLabelFile = os.path.join("/hdd4", valLabelFile)
#         newTestFeatureFile = os.path.join("/hdd4", newTestFeatureFile)

    
#     createBatchTrainingParams()
#     batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
    


#     train_total = batchTrainingParams[modelname]['generate_train_total']


#     verbose = 0;
#     for e in range(epochs):
        
#         for subEpochNumber in range(train_total):
#             verbose = 0

            
#             if(subEpochNumber%2 == 0):
#                 print("epoch %d / %d" %(e,epochs-1))
#                 print("subepoch %d / %d" %(subEpochNumber,train_total-1))
#                 verbose = 1
            
#             xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
#                                                 trainFeatureFile,
#                                                  trainLabelFile, numofFiles = float('inf'))
#             xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
#                                             valFeatureFile,
#                                          valLabelFile, numofFiles = float('inf'))



#             hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
#                                 verbose = verbose, 
#                             validation_data = (xVal,yVal), callbacks = None,
#                              class_weight = class_weight)




#     print(hist.history)

#     print("Done Training!")
#     print("Fetching Predictions ....")



    
#     predictions = np.empty((0,1))
#     iterations = batchTrainingParams[modelname]['predict_iterations']
#     for i in range(iterations):
#         dataToPredictOn = generate_pred_new(i,modelname, featureParams,
#                                      event, valFeatureFile)
#         #print(model.predict(dataToPredictOn).shape)
#         temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

#         predictions = np.vstack((predictions, temp )).reshape((-1,1)) 


#     predictions2 = np.empty((0,1))
#     iterations = batchTrainingParams[modelname]['newTest_iterations']
#     for i in range(iterations):
#         dataToPredictOn = generate_pred_new(i,modelname, featureParams,
#                                      event, newTestFeatureFile)
#         #print(model.predict(dataToPredictOn).shape)
#         temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

#         predictions2 = np.vstack((predictions2, temp )).reshape((-1,1))       


#     print("final predictions.shape", predictions2.shape)

#     return predictions,hist.history


def train_and_predict_dnn_save(modelname,event,modelParams, directory,
                      numTrainFiles = float('inf'), numValidfiles = float('inf'), index=None, use_input_model_file = False
                      ,input_model_file = None):

    #createParamsFile()


    modelParams = loadfeatureParams('modelParams.yaml')

    if modelParams[modelname][event]['load_model'] == 'new':
        model = DNN(modelParams,modelname,event)
    elif (modelParams[modelname][event]['load_model'] == 'load' and (use_input_model_file == False) ):
        print("Loading model from file...", modelParams[modelname][event]['model_path'])
        model = load_model(modelParams[modelname][event]['model_path'])
    elif use_input_model_file:
        print("Loading model from file...", input_model_file)
        model = load_model(input_model_file)





    print ("Priniting Model Architecture\n ")
    model.summary()
    print( "Done printing model summary! \n")

    trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

    trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

    valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

    valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

    newTestFeatureFile = directory+ '_' + event + '_' + 'newTest' + '_features.h5'

    evalFeatureFile = directory+ '_' + event + '_' + 'eval' + '_features.h5'

    if modelParams['mycomp']:
        trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
        trainLabelFile = os.path.join("../../../../../", trainLabelFile)
        valFeatureFile = os.path.join("../../../../../", valFeatureFile)
        valLabelFile = os.path.join("../../../../../", valLabelFile)
        newTestFeatureFile = os.path.join("../../../../../", newTestFeatureFile)
        evalFeatureFile = os.path.join("../../../../../", evalFeatureFile)
    else:
        trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
        trainLabelFile = os.path.join("/hdd4", trainLabelFile)
        valFeatureFile = os.path.join("/hdd4", valFeatureFile)
        valLabelFile = os.path.join("/hdd4", valLabelFile)
        newTestFeatureFile = os.path.join("/hdd4", newTestFeatureFile)
        evalFeatureFile = os.path.join("/hdd4", evalFeatureFile)


    #batch_size=modelParams[modelname][event]['batch_size']
    #epochs= modelParams[modelname][event]['epochs']
    
    #validation_data = (xVal,yVal)

    if modelParams[modelname][event]['train_model']:

        min_delta = modelParams['min_delta']
        patience = modelParams['patience']
        earlystopping = EarlyStopping(monitor='val_loss',
                         min_delta=min_delta, patience=patience, verbose=0, mode='auto')

        batch_size=modelParams[modelname][event]['batch_size']
        epochs = modelParams[modelname][event]['epochs']

        myEpochs = [2,3,4,2,3,4]
        if modelParams[modelname][event]['override_epochs']:
            epochs = myEpochs[index]


        class_weight = None
        if modelParams['class_weights_enable'] ==True:
            class_weight = modelParams[modelname][event]['class_weights']

        #print(class_weight)
        #sys.exit()




        
        
        createBatchTrainingParams()
        batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
        
        #train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
        #val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']

        # for e in range(nb_epoch):
        # print("epoch %d" % e)
        # for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        #     model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)

        train_total = batchTrainingParams[modelname]['generate_train_total']



        verbose = 0;
        for e in range(epochs):
            
            for subEpochNumber in range(train_total):
                verbose = 0

                
                if(subEpochNumber%20 == 0):
                    print("epoch %d / %d" %(e,epochs-1))
                    print("subepoch %d / %d" %(subEpochNumber,train_total-1))
                    verbose = 1
                
                xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
                                                    trainFeatureFile,
                                                     trainLabelFile, numofFiles = float('inf'))
                xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
                                                valFeatureFile,
                                             valLabelFile, numofFiles = float('inf'))



                hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
                                    verbose = verbose, 
                                validation_data = (xVal,yVal), callbacks = None,
                                 class_weight = class_weight)





   
        print(hist.history)

        print("Done Training!")
        print("Fetching Predictions ....")



    if modelParams[modelname][event]['save_model']:
        fileName = directory+ '_' + event + '_' + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y")+  '_ModelFile.h5' 
        model.save(fileName)
        print("Saved model to: ", fileName)


    createBatchTrainingParams()
    batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
   
    print("fetching VAL predictions...")
    
    predictions = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['predict_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, valFeatureFile)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions = np.vstack((predictions, temp )).reshape((-1,1)) 

    print("fetching TEST predictions...")
    predictions2 = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['newTest_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, newTestFeatureFile, flag = True)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions2 = np.vstack((predictions2, temp )).reshape((-1,1))       


    if modelParams[modelname][event]['eval_predict']:
        print("Fetching EVAL_predictions...")
        predictions_eval = np.empty((0,1))
        iterations = batchTrainingParams[modelname]['eval_iterations']
        for i in range(iterations):
            dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                         event, evalFeatureFile, mode = 'eval')
            #print(model.predict(dataToPredictOn).shape)
            temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

            predictions_eval = np.vstack((predictions_eval, temp )).reshape((-1,1))      

        print("final predictions_EVAL.shape", predictions_eval.shape)


    print("final predictions.shape", predictions.shape)

    print("final predictions2.shape", predictions2.shape)

    

    if modelParams[modelname][event]['eval_predict']:
          return predictions, predictions2, predictions_eval, 0

    if modelParams[modelname][event]['train_model']:
        return predictions, predictions2, 0, hist.history
    else:
        return predictions, predictions2,0,0


def train_and_predict_cnn_save(modelname,event,modelParams, directory,
                      numTrainFiles = float('inf'), numValidfiles = float('inf'), index=None, use_input_model_file = False
                      ,input_model_file = None):

    modelParams = loadfeatureParams('modelParams.yaml')

    if modelParams[modelname][event]['load_model'] == 'new':
        model = CNN(modelParams,modelname,event)
    elif modelParams[modelname][event]['load_model'] == 'load' and (use_input_model_file == False):
        print("Loading model from file...", modelParams[modelname][event]['model_path'])
        model = load_model(modelParams[modelname][event]['model_path'])
    elif use_input_model_file:
        print("Loading model from file...", input_model_file)
        model = load_model(input_model_file)



    print ("Priniting Model Architecture\n ")
    model.summary()
    print( "Done printing model summary! \n")

    trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

    trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

    valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

    valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

    newTestFeatureFile = directory+ '_' + event + '_' + 'newTest' + '_features.h5'

    evalFeatureFile = directory+ '_' + event + '_' + 'eval' + '_features.h5'

    if modelParams['mycomp']:
        trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
        trainLabelFile = os.path.join("../../../../../", trainLabelFile)
        valFeatureFile = os.path.join("../../../../../", valFeatureFile)
        valLabelFile = os.path.join("../../../../../", valLabelFile)
        newTestFeatureFile = os.path.join("../../../../../", newTestFeatureFile)
        evalFeatureFile = os.path.join("../../../../../", evalFeatureFile)
    else:
        trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
        trainLabelFile = os.path.join("/hdd4", trainLabelFile)
        valFeatureFile = os.path.join("/hdd4", valFeatureFile)
        valLabelFile = os.path.join("/hdd4", valLabelFile)
        newTestFeatureFile = os.path.join("/hdd4", newTestFeatureFile)
        evalFeatureFile = os.path.join("/hdd4", evalFeatureFile)


    #batch_size=modelParams[modelname][event]['batch_size']
    #epochs= modelParams[modelname][event]['epochs']
    
    #validation_data = (xVal,yVal)

    if modelParams[modelname][event]['train_model']:

        min_delta = modelParams['min_delta']
        patience = modelParams['patience']
        earlystopping = EarlyStopping(monitor='val_loss',
                         min_delta=min_delta, patience=patience, verbose=0, mode='auto')

        batch_size=modelParams[modelname][event]['batch_size']
        epochs = modelParams[modelname][event]['epochs']

        myEpochs = [2,3,4,2,3,4]
        if modelParams[modelname][event]['override_epochs']:
            epochs = myEpochs[index]


        class_weight = None
        if modelParams['class_weights_enable'] ==True:
            class_weight = modelParams[modelname][event]['class_weights']

        #print(class_weight)
        #sys.exit()




        
        
        createBatchTrainingParams()
        batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
        
        #train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
        #val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']

        # for e in range(nb_epoch):
        # print("epoch %d" % e)
        # for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        #     model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)

        train_total = batchTrainingParams[modelname]['generate_train_total']



        verbose = 0;
        for e in range(epochs):
            
            for subEpochNumber in range(train_total):
                verbose = 0

                
                if(subEpochNumber%20 == 0):
                    print("epoch %d / %d" %(e,epochs-1))
                    print("subepoch %d / %d" %(subEpochNumber,train_total-1))
                    verbose = 1
                
                xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
                                                    trainFeatureFile,
                                                     trainLabelFile, numofFiles = float('inf'))
                xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
                                                valFeatureFile,
                                             valLabelFile, numofFiles = float('inf'))



                hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
                                    verbose = verbose, 
                                validation_data = (xVal,yVal), callbacks = None,
                                 class_weight = class_weight)





   
        print(hist.history)

        print("Done Training!")
        print("Fetching Predictions ....")



    if modelParams[modelname][event]['save_model']:
        fileName = directory+ '_' + event + '_' + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y")+  '_ModelFile.h5' 
        model.save(fileName)
        print("Saved model to: ", fileName)


    createBatchTrainingParams()
    batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
   

    print("fetching VAL predictions...")
    predictions = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['predict_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, valFeatureFile)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions = np.vstack((predictions, temp )).reshape((-1,1)) 

    print("fetching TEST predictions...")
    predictions2 = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['newTest_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, newTestFeatureFile, flag = True)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions2 = np.vstack((predictions2, temp )).reshape((-1,1))       


    if modelParams[modelname][event]['eval_predict']:
        print("Fetching EVAL_predictions...")
        predictions_eval = np.empty((0,1))
        iterations = batchTrainingParams[modelname]['eval_iterations']
        for i in range(iterations):
            dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                         event, evalFeatureFile, mode = 'eval')
            #print(model.predict(dataToPredictOn).shape)
            temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

            predictions_eval = np.vstack((predictions_eval, temp )).reshape((-1,1))      

        print("final predictions_EVAL.shape", predictions_eval.shape)


    print("final predictions.shape", predictions.shape)

    print("final predictions2.shape", predictions2.shape)

    

    if modelParams[modelname][event]['eval_predict']:
          return predictions, predictions2, predictions_eval, 0

    if modelParams[modelname][event]['train_model']:
        return predictions, predictions2, 0, hist.history
    else:
        return predictions, predictions2,0,0

def train_and_predict_rnn_save(modelname,event,modelParams, directory,
                      numTrainFiles = float('inf'), numValidfiles = float('inf'), index=None, use_input_model_file = False
                      ,input_model_file = None):

    modelParams = loadfeatureParams('modelParams.yaml')

    if modelParams[modelname][event]['load_model'] == 'new':
        model = RNN(modelParams,modelname,event)
    elif modelParams[modelname][event]['load_model'] == 'load' and (use_input_model_file == False):
        print("Loading model from file...", modelParams[modelname][event]['model_path'])
        model = load_model(modelParams[modelname][event]['model_path'])
    elif use_input_model_file:
        print("Loading model from file...", input_model_file)
        model = load_model(input_model_file)


    print ("Priniting Model Architecture\n ")
    model.summary()
    print( "Done printing model summary! \n")

    trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

    trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

    valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

    valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

    newTestFeatureFile = directory+ '_' + event + '_' + 'newTest' + '_features.h5'

    evalFeatureFile = directory+ '_' + event + '_' + 'eval' + '_features.h5'

    if modelParams['mycomp']:
        trainFeatureFile = os.path.join("../../../../../", trainFeatureFile)
        trainLabelFile = os.path.join("../../../../../", trainLabelFile)
        valFeatureFile = os.path.join("../../../../../", valFeatureFile)
        valLabelFile = os.path.join("../../../../../", valLabelFile)
        newTestFeatureFile = os.path.join("../../../../../", newTestFeatureFile)
        evalFeatureFile = os.path.join("../../../../../", evalFeatureFile)
    else:
        trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
        trainLabelFile = os.path.join("/hdd4", trainLabelFile)
        valFeatureFile = os.path.join("/hdd4", valFeatureFile)
        valLabelFile = os.path.join("/hdd4", valLabelFile)
        newTestFeatureFile = os.path.join("/hdd4", newTestFeatureFile)
        evalFeatureFile = os.path.join("/hdd4", evalFeatureFile)


    #batch_size=modelParams[modelname][event]['batch_size']
    #epochs= modelParams[modelname][event]['epochs']
    
    #validation_data = (xVal,yVal)

    if modelParams[modelname][event]['train_model']:

        min_delta = modelParams['min_delta']
        patience = modelParams['patience']
        earlystopping = EarlyStopping(monitor='val_loss',
                         min_delta=min_delta, patience=patience, verbose=0, mode='auto')

        batch_size=modelParams[modelname][event]['batch_size']
        epochs = modelParams[modelname][event]['epochs']

        myEpochs = [2,3,4,2,3,4]
        if modelParams[modelname][event]['override_epochs']:
            epochs = myEpochs[index]


        class_weight = None
        if modelParams['class_weights_enable'] ==True:
            class_weight = modelParams[modelname][event]['class_weights']

        #print(class_weight)
        #sys.exit()




        
        
        createBatchTrainingParams()
        batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
        
        #train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
        #val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']

        # for e in range(nb_epoch):
        # print("epoch %d" % e)
        # for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        #     model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)

        train_total = batchTrainingParams[modelname]['generate_train_total']



        verbose = 0;
        for e in range(epochs):
            
            for subEpochNumber in range(train_total):
                verbose = 0

                
                if(subEpochNumber%20 == 0):
                    print("epoch %d / %d" %(e,epochs-1))
                    print("subepoch %d / %d" %(subEpochNumber,train_total-1))
                    verbose = 1
                
                xTrain,yTrain = generate_train_new(subEpochNumber, modelname, featureParams, event,
                                                    trainFeatureFile,
                                                     trainLabelFile, numofFiles = float('inf'))
                xVal,yVal = generate_val_new(subEpochNumber, modelname, featureParams, event, 
                                                valFeatureFile,
                                             valLabelFile, numofFiles = float('inf'))



                hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = 1,
                                    verbose = verbose, 
                                validation_data = (xVal,yVal), callbacks = None,
                                 class_weight = class_weight)





   
        print(hist.history)

        print("Done Training!")
        print("Fetching Predictions ....")



    if modelParams[modelname][event]['save_model']:
        fileName = directory+ '_' + event + '_' + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y")+  '_ModelFile.h5' 
        model.save(fileName)
        print("Saved model to: ", fileName)


    createBatchTrainingParams()
    batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
   

    print("fetching VAL predictions...")
    predictions = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['predict_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, valFeatureFile)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions = np.vstack((predictions, temp )).reshape((-1,1)) 

    print("fetching TEST predictions...")
    predictions2 = np.empty((0,1))
    iterations = batchTrainingParams[modelname]['newTest_iterations']
    for i in range(iterations):
        dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                     event, newTestFeatureFile,flag = True)
        #print(model.predict(dataToPredictOn).shape)
        temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

        predictions2 = np.vstack((predictions2, temp )).reshape((-1,1))       


    if modelParams[modelname][event]['eval_predict']:
        print("Fetching EVAL_predictions...")
        predictions_eval = np.empty((0,1))
        iterations = batchTrainingParams[modelname]['eval_iterations']
        for i in range(iterations):
            dataToPredictOn = generate_pred_new(i,modelname, featureParams,
                                         event, evalFeatureFile, mode = 'eval')
            #print(model.predict(dataToPredictOn).shape)
            temp  = np.array(model.predict(dataToPredictOn)).reshape((-1,1)) 

            predictions_eval = np.vstack((predictions_eval, temp )).reshape((-1,1))      

        print("final predictions_EVAL.shape", predictions_eval.shape)


    print("final predictions.shape", predictions.shape)

    print("final predictions2.shape", predictions2.shape)

    

    if modelParams[modelname][event]['eval_predict']:
          return predictions, predictions2, predictions_eval, 0

    if modelParams[modelname][event]['train_model']:
        return predictions, predictions2, 0, hist.history
    else:
        return predictions, predictions2,0,0
# def train_and_predict_cnn(modelname,event,modelParams, directory,
#                       numTrainFiles = float('inf'), numValidfiles = float('inf')):

    
#     #createParamsFile()
#     modelParams = loadfeatureParams('modelParams.yaml')
#     model = CNN(modelParams,modelname,event)



#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


    
#     #batch_size=modelParams[modelname][event]['batch_size']
#     #epochs= modelParams[modelname][event]['epochs']
    
#     #validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']
#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

    

#     trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

#     trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

#     valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

#     valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

#     trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
#     trainLabelFile = os.path.join("/hdd4", trainLabelFile)
#     valFeatureFile = os.path.join("/hdd4", valFeatureFile)
#     valLabelFile = os.path.join("/hdd4", valLabelFile)
#     epochs = modelParams[modelname][event]['epochs']
    
#     createBatchTrainingParams()
#     batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
#     train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
#     val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']
    


#     hist = model.fit_generator(generate_train('cnn', featureParams, event, trainFeatureFile, trainLabelFile,

#                                  numofFiles = numTrainFiles),
#                                steps_per_epoch=train_steps,
#                                epochs=epochs,verbose=1,
#                                callbacks = [earlystopping],
#                                validation_data=generate_val('cnn', featureParams,  event, valFeatureFile,
#                                                     valLabelFile, 
#                                                     numofFiles = numValidfiles),
#                                validation_steps=val_steps,
#                                class_weight=class_weight
#                                 )

#     # hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
#     #              validation_data = validation_data, callbacks = [earlystopping], class_weight = class_weight)

#     print(hist.history)

#     print("Done Training!")
#     print("Fetching Predictions ....")


#     predictions = model.predict_generator( generate_pred('cnn', featureParams, event, valFeatureFile),
#                                              steps=val_steps, verbose=1)
    


#     return predictions,hist.history



# def train_and_predict_dnn(modelname,event,modelParams, directory,
#                       numTrainFiles = float('inf'), numValidfiles = float('inf')):

#     #createParamsFile()
#     modelParams = loadfeatureParams('modelParams.yaml')
#     model = DNN(modelParams,modelname,event)



#     print ("Priniting Model Architecture\n ")
#     model.summary()
#     print( "Done printing model summary! \n")


    
#     #batch_size=modelParams[modelname][event]['batch_size']
#     #epochs= modelParams[modelname][event]['epochs']
    
#     #validation_data = (xVal,yVal)

#     min_delta = modelParams['min_delta']
#     patience = modelParams['patience']
#     earlystopping = EarlyStopping(monitor='val_loss',
#                      min_delta=min_delta, patience=patience, verbose=0, mode='auto')

#     class_weight = None
#     if modelParams['class_weights_enable'] ==True:
#         class_weight = modelParams[modelname][event]['class_weights']

#     print(class_weight)
#     #sys.exit()

#     trainFeatureFile = directory+ '_' + event + '_' + 'devtrain' + '_features.h5'

#     trainLabelFile = directory+ '_' + event + '_' +  'devtrain' + '_labels.h5'

#     valFeatureFile = directory+ '_' + event + '_' + 'devtest' + '_features.h5'

#     valLabelFile = directory+ '_' + event + '_' +  'devtest' + '_labels.h5'

#     trainFeatureFile = os.path.join("/hdd4", trainFeatureFile)
#     trainLabelFile = os.path.join("/hdd4", trainLabelFile)
#     valFeatureFile = os.path.join("/hdd4", valFeatureFile)
#     valLabelFile = os.path.join("/hdd4", valLabelFile)
#     epochs = modelParams[modelname][event]['epochs']
    
#     createBatchTrainingParams()
#     batchTrainingParams  = loadfeatureParams('batchTrainingParams.yaml')
#     train_steps = batchTrainingParams[modelname]['training_steps_per_epoch']
#     val_steps = batchTrainingParams[modelname]['val_steps_per_epoch']



#     hist = model.fit_generator(generate_train('dnn', featureParams, event, trainFeatureFile,
#                                      trainLabelFile
#                                  ),
#                                steps_per_epoch=train_steps,
#                                epochs=epochs,verbose=1,
#                                callbacks = [earlystopping],
#                                validation_data=generate_val('dnn', featureParams, event, valFeatureFile,
#                                                         valLabelFile
#                                                     ),
#                                validation_steps=val_steps,
#                                class_weight=None
#                                 )

#     temp = model.layers[0]
#     print(temp.get_weights())

#     # hist = model.fit(xTrain,yTrain,batch_size = batch_size, epochs = epochs, verbose = 1, 
#     #              validation_data = validation_data, callbacks = [earlystopping], class_weight = class_weight)

#     print(hist.history)

#     print("Done Training!")
#     print("Fetching Predictions ....")


#     predictions = model.predict_generator( generate_pred('dnn', featureParams, event, valFeatureFile),
#                                                 steps=1, verbose=1)
    


#     return predictions,hist.history
