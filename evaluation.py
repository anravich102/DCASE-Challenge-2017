import numpy as np 
import os,sys
from createParams import *







def process_predictions(predictions, model, directory, featureParams, modelParams,
                         event, numFramesList, filesList):

    '''numFrames is the number of samples/images generated by a single audio file '''
    ''' directory is the model + timestamp string '''
    
    '''numFramesList,filesList must be passed from while loading test/evaluation data'''

    files = []
    startTimes = []
    endTimes = []

    index = 0
    for file,numFramesInFile in zip(filesList,numFramesList):
        
        files.append(file)
        predictionsForFile = predictions[index:numFramesInFile]
        index = index + numFramesInFile
        startTime,endTime = get_times(predictionsForFile,file,numFramesInFile, event, model,
                                     modelParams, featureParams)

        startTimes.append(startTime)
        endTimes.append(endTime)

    annotatedFile = write_output_file(files, startTimes, endTimes, directory,event)

  
    return annotatedFile


def get_times(predictionsForFile,file,numFramesInFile, event, model,
                                     modelParams, featureParams):
    
    
    # pass predictionForFile as an array 
    processedPredictions = post_process(predictionsForFile)

    pp = processedPredictions

    #pp is a bunch of zeros or ones only, with a sungle chunk of ones.

    if model=='dnn':
        if np.array_equal(pp, np.zeros(pp.shape)):
            startTime = -1
            endTime = -1
        else:
            y = np.where(x == 1)

            index = y[0][0]
            endIndex = y[0][-1]
            startTime = index * ((100.0 - modelParams[model][event]['percent_overlap'])/100.0)*\
                            modelParams[model][event]['win_length']

            endTime = endIndex * ((100.0 - modelParams[model][event]['percent_overlap'])/100.0)*\
                            modelParams[model][event]['win_length']


 
    
    return startTime, endTime

import scipy.signal
def post_process(predictionsForFile):
    filtered = scipy.signal.medfilt(x, kernel_size=3)
    filtered = filtered.tolist()
    pp = rounded = [round(x[0]) for x in filtered]
    return pp 



def write_output_file(files, startTimes, endTimes , directory ,event):

    '''start time contains -1 if event did not occur'''
    '''end time contains final window timestamp if event continues till end'''
    

    #first make the required directories:
    resultsDir = os.path.join(os.getcwd(), '../Results')
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)

    # save parameters  of the currrent model (using same name as  directory name) for reference later:

    annotatedFile = directory + '_' + event + '.csv'

    pathToSave = os.path.join(resultsDir,annotatedFile)

    with open(pathToSave,'w') as f:
        for file, start, end in zip(files, startTimes,endTimes):
            if(start != -1):
                f.write(file + "\t" + start + "\t" + end+ "\t" + event + "\n")
            else:
                f.write(file +  "\n")


    return pathToSave


def evaluate(userAnnotatedFile, groundTruthFile, event):

    ins = 0
    dele = 0
    fp = fn = tp = 0.0
    total = 0.0

    with open(userAnnotatedFile) as userfile, open(groundTruthFile) as truthfile: 
        for x, y in zip(userfile, truthfile):

            userline = x.strip().split()
            truthline = y.strip().split()


            if(len(userline) > len(truthline) ):
                fp += 1
                ins += 1

            if( len(truthline) > len(userline)):
                fn+=1
                dele+=1
                if(len(truthline) == 4):
                    total += 1

            if( len(truthline) == len(userline) and len(truthline) == 4):
                total+=1
                #if event within collar
                collar = abs(float(userline[1]) - float(truthline[1]))
                if collar <= 0.5 and userline[3] == truthline[3]:
                    tp += 1


                #not within collar
                else:
                    fp+=1
                    fn+=1
                    ins+=1
                    dele+=1


    er = (ins+dele)/total
    Fscore = 2*tp/ (2*tp + fp + fn)


    return er, Fscore


userAnnotatedFile = 'f1.txt'
groundTruthFile = 'f2.txt'

evaluate(userAnnotatedFile,groundTruthFile,'babycry')
