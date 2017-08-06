import numpy as np 
import os,sys
from PARAMS import *
from POSTPROCESS import *
import random







def simulate(filepath):
    d = loadfeatureParams(filepath)
    print("simulating PP parameters...")

    modelParams = loadfeatureParams('modelParams.yaml')

    erList = []
    fscoreList = []
    medfiltList = []
    thresholdList = []

    print(modelParams['pp_tuning_countmax'])

    for i in range(modelParams['pp_tuning_countmax']):

        kernel_size = random.randint(2,13)
        if kernel_size%2 == 0:
            kernel_size = kernel_size+1
        clasify_threshold = random.uniform(0.3,0.8)

        model = d['model']
        event = d['event']

        modelParams[model][event]['medfit_kernel'] = kernel_size
        modelParams[model][event]['classify_threshold'] = clasify_threshold

        DumpfeatureParams(modelParams,'modelParams.yaml')

        userAnnotatedFile = process_predictions(d['predictions'], d['model'], d['directory'], d['featureParams'],
                                                 modelParams,
                                                 d['event'], d['numFramesList'], d['fileList'], mode = 'val') #produce files



        pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
        listofFiles  = os.listdir(pathToFile) 
        filename = 'event_list_devtest_'+event +'.csv'
        groundTruthFile = os.path.join(pathToFile, filename)

        er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
        erList.append(er)
        fscoreList.append(fscore)
        medfiltList.append(kernel_size)
        thresholdList.append(clasify_threshold)

        print(str(i)+"/"+str(modelParams['pp_tuning_countmax']))


    print("Fscore ERate KernelSize, Classify _hreshold")
    for item in sorted(zip(fscoreList, erList,medfiltList, thresholdList )):
        print(item)
    #print ((sorted(zip(fscoreList, erList,medfiltList, thresholdList ))) )



def simulate_single_run(filepath):
    d = loadfeatureParams(filepath)
    print("simulating PP parameters...")

    modelParams = loadfeatureParams('modelParams.yaml')


    model = d['model']
    event = d['event']

    DumpfeatureParams(modelParams,'modelParams.yaml')

    userAnnotatedFile = process_predictions(d['predictions'], d['model'], d['directory'], d['featureParams'],
                                             modelParams,
                                             d['event'], d['numFramesList'], d['fileList']) #produce files



    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtest_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics

    print("Medfilt Kernel_size: ", modelParams[model][event]['medfilt_kernel'],
             "Classify Threshold:", modelParams[model][event]['classify_threshold'])
    
    print("ER:", er, "Fscore:", fscore, "fp:", fp, "fn:",fn, "ins:", ins, "Dele:", dele, 'tp:', tp)
    
   
if __name__ == "__main__":


    event = raw_input('Enter the event name: ')
    if event == 'babycry':    
        file = 'rnn_01_57_PM_July_31_2017babycry_raw_Predictions.h5'
    elif event == 'glassbreak':
        file = 'rnn_01_57_PM_July_31_2017glassbreak_raw_Predictions.h5'
    elif event == 'gunshot':
        file = 'rnn_01_57_PM_July_31_2017gunshot_raw_Predictions.h5'

    createParamsFile()

    modelParams = loadfeatureParams('modelParams.yaml')
    if modelParams['simulate_single_run'] == True:
        simulate_single_run(file)
    else:
        simulate(file)
