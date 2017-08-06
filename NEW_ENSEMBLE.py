 
import datetime
import os
import numpy as np
import sys

from POSTPROCESS import get_metrics

def get_results(userAnnotatedFile,userAnnotatedFile2, event):

    
    print("VAL SET RESULTS:")
    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/devtest/20b255387a2d0cddc0a3dff5014875e7/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtest_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile, groundTruthFile, event) #produce metrics
    

    print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
   
    print("event: ", event)
    print("er :",er )
    print("Fscore :",fscore)

    






    print("TEST SET RESULTS:")

    pathToFile = os.path.join(os.getcwd(), '../data/mixture_data/newTest/81fc1201069905c5926c9c5c8369eacf/meta')
    listofFiles  = os.listdir(pathToFile) 
    filename = 'event_list_devtrain_'+event +'.csv'
    groundTruthFile = os.path.join(pathToFile, filename)

    er, fscore,fp,fn,ins,dele,tp= get_metrics(userAnnotatedFile2, groundTruthFile, event) #produce metrics
   
    print("fp:",fp,"fn:",fn, "ins:", ins, "dele", dele, "tp:", tp)
    print("event: ", event)
    print("er :",er )
    print("Fscore :",fscore)

    



                                                #weights in descending order
def ensemble_late_fusion(ListofFiles, weights, event, mode):


    #read one file to get list of files
    #read alll files to get start times,end times

    for index, name in enumerate(ListofFiles):
       
        ListofFiles[index] =  os.path.join(os.getcwd(), '../Results', str(name+'.csv'))

    

    filesList = []
    startTimeList = []
    endTimeList = []
    event_present = []

    #normalize weights:

    weight_sum = sum(weights)
    weights = [(x/weight_sum) for x in weights]



    openfileList = [open(path, 'r') for path in ListofFiles]
    #print(len(openfileList))

    with open(ListofFiles[0],'r') as f:
        for line in f:
            line1 = line
            userline = line1.strip().split()
            filesList.append(userline[0])

    #print(openfileList)
    #print("open" , len(openfileList))

    print("length of filesList: ", len(filesList))

    for name  in ListofFiles:

        with open(name, 'r') as f:

        
            temp_eventPresent = []
            temp_start = []
            temp_end = []

         

           
            for line in f:

                line1 = line
                userline = line1.strip().split()


                if(len(userline) == 1 or len(userline) == 2 or len(userline) == 3 ):
                    temp_eventPresent.append(0)
                    
                    temp_start.append(-1)
                    temp_end.append(-1)

                elif(len(userline) == 4): 
                    
                    temp_eventPresent.append(1)  
                    temp_start.append(float(userline[1]))
                    temp_end.append(float(userline[2]))

           
            startTimeList.append(temp_start)
            endTimeList.append(temp_end)
            event_present.append(temp_eventPresent)
        
    
    assert (len(event_present[0]) == 500  and len(event_present[1]) == 500 and len(startTimeList[0]) == 500)

    # now use the weights to ensemble 

    ensembledStartTimes = []
    ensembledEndTimes = []



    for evalFileIndex, evalFile in enumerate(filesList):

        #first determine if the event has occured

        event_present_flag = 0
        for j,userfile in enumerate(ListofFiles):
            

            event_present_flag +=  (weights[j]*event_present[j][evalFileIndex])

        if(event_present_flag < 0.5):
            ensembledStartTimes.append(-1)
            ensembledEndTimes.append(-1)
        else:

            #if there is an event present. replicate the results of the model with highest weight that thinks there is an event

            for j,userfile in enumerate(ListofFiles):

                if event_present[j][evalFileIndex] == 0:
                    continue
                else:

                    ensembledStartTimes.append(startTimeList[j][evalFileIndex])
                    ensembledEndTimes.append(endTimeList[j][evalFileIndex])
                    break


    annotatedFile = event + '_' + mode + '_' + datetime.datetime.now().strftime("%I_%M_%p_%B_%d_%Y") + '.csv'
    
    mainList = [0]*500

    for file, start, end in zip(filesList, ensembledStartTimes,ensembledEndTimes):
        index =  int(file.split("_")[3])
        
        
        mainList[index] = [file, start,end]



    with open(annotatedFile,'w') as f:
        for item in mainList:

            if item == 0:
                continue

            file = item[0]
            start = float(item[1])
            end = float(item[2])
            if(start != -1.0):
                f.write(file + "\t" + str(round(start, 4)) + "\t" + str(round(end, 4)) + "\t" + event + "\n")
            else:
                f.write(file +  "\n")
            

    print("Saved file to: ", annotatedFile)

    return annotatedFile



### VAL


ListofFiles = ['dnn_10_06_PM_August_02_2017_glassbreak_VAL_09_09_PM_August_05_2017', 
                'dnn_10_06_PM_August_02_2017_glassbreak_VAL_09_05_PM_August_05_2017',
                'dnn_10_06_PM_August_02_2017_glassbreak_VAL_09_01_PM_August_05_2017']
weights = [0.5,0.25,0.25]
event  = 'glassbreak'
mode = 'val'

userAnnotatedFile  =  ensemble_late_fusion(ListofFiles, weights, event, mode)



### TEST


ListofFiles = ['dnn_10_06_PM_August_02_2017_glassbreak_TEST_09_09_PM_August_05_2017',
                'dnn_10_06_PM_August_02_2017_glassbreak_TEST_09_02_PM_August_05_2017',
                'dnn_10_06_PM_August_02_2017_glassbreak_TEST_08_16_PM_August_05_2017']

mode = 'test'

userAnnotatedFile2  =  ensemble_late_fusion(ListofFiles, weights, event, mode)




get_results(userAnnotatedFile,userAnnotatedFile2, event)

