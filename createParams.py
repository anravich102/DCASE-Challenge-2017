
import yaml
import os



dnnbabycryParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}

dnnglassbreakParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}

dnngunshotParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}




dnnParams = { 'babycry' : dnnbabycryParams,
              'glassbreak' : dnnglassbreakParams,
              'gunshot' : dnngunshotParams}




cnnbabycryParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True,
                    'step': 8, 
                    'label_group_threshold': 0.6,
                    'type': 'logmagspec'} #type can be 'magspec','powerspec',
                                       #'logmagspec' or 'logpowerspec'

cnnglassbreakParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True,
                    'step': 8,
                    'label_group_threshold': 0.6,
                    'type': 'logmagspec'
                    }

cnngunshotParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True,
                    'step': 8,
                    'label_group_threshold': 0.6, 
                    'type': 'logmagspec'}


cnnParams = { 'babycry' : cnnbabycryParams,
              'glassbreak' : cnnglassbreakParams,
              'gunshot' : cnngunshotParams}


source_babycryParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}
                    
source_glassbreakParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}
                    
source_gunshotParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'doubledelta': True}

source_dataParams = {'babycry': source_babycryParams,
                    'glassbreak': source_glassbreakParams,
                    'gunshot': source_gunshotParams }
                    

d1 = {'dnn': dnnParams, 
      'cnn' : cnnParams,
      'source_data' : source_dataParams }





def loadfeatureParams(filepath):
    ''' returns: Feature Parmas Dict'''

    stream = open(filepath, 'r')
    return yaml.load(stream)


def DumpfeatureParams(Paramdict, filename):
    ''' returns: Path to FeatureParmas.yaml'''

    

    filepath = os.path.join(os.getcwd(), filename)
    stream = open(filepath, 'w')
    yaml.dump(Paramdict, stream)
    return filepath

#file1 = 'FeatureParams.yaml'
#yamlfile = DumpfeatureParams(d1)

#featureParams = loadfeatureParams(yamlfile)

#print(featureParams)


dnnbabycryParams = { 'num_layers': 3,
              'batch_norm': [True,True,True],
              'num_units':[256,128,32],
              'Dropout': [0.5,0.3,0.3],
               'lr': 0.001,
               'epochs': 2,
               'batch_size': 20,
               'reg': 0.0,
                }


dnnglassbreakParams = { 'num_layers': 3,
              'BatchNorm': [True,True,True],
              'num_units':[256,128,32],
              'Dropout': [0.5,0.3,0.3],
               'lr': 0.001,
               'epochs': 2,
               'batch_size': 20,
               'reg': 0.0 }


dnngunshotParams = { 'num_layers': 3,
              'BatchNorm': [True,True,True],
              'num_units':[256,128,32],
              'Dropout': [0.5,0.3,0.3],
               'lr': 0.001,
               'epochs': 2,
               'batch_size': 20,
               'reg': 0.0 }





dnnParams = {'babycry': dnnbabycryParams,
             'glassbreak': dnnglassbreakParams,
             'gunshot': dnngunshotParams }




cnnbabycryParams = {'BatchNorm': True, 
                    'l2':0.0,
                    'xdim': 8,
                    'ydim' : 1025,
                    'num_channels': 1
              }


cnnglassbreakParams = {'BatchNorm': True, 
                    'l2':0.0,
                    'xdim': 8,
                    'ydim' : 1025,
                    'num_channels': 1
              }


cnngunshotParams = {'BatchNorm': True, 
                    'l2':0.0,
                    'xdim': 8,
                    'ydim' : 1025,
                    'num_channels': 1
              }


cnnParams = {'babycry':cnnbabycryParams,
             'glassbreak': cnnglassbreakParams,
             'gunshot': cnngunshotParams }


modelParams = { 'dnn': dnnParams, 
                'cnn' : cnnParams,
                }

yamlfile = DumpfeatureParams(modelParams,'modelParams.yaml')

modelParamsReloaded = loadfeatureParams(yamlfile)

print(modelParamsReloaded)

