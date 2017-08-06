
import yaml
import os



dnnbabycryParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 60,
                    'delta':True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}

dnnglassbreakParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 60,
                    'delta': True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}

dnngunshotParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels': 64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 60,
                    'delta': True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}






dnnParams = { 'babycry' : dnnbabycryParams,
              'glassbreak' : dnnglassbreakParams,
              'gunshot' : dnngunshotParams}




cnnbabycryParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.1,
                    'percent_overlap': 80,
                    'n_mfcc' : 64,
                    'delta': False,
                    'doubledelta': False,
            
                    'label_type': 'normal' ,#'normal'
                    'type': None }  #'logmagspec'     #type can be 'magspec','powerspec',
                                       #'logmagspec' or 'logpowerspec'

cnnglassbreakParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':128,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.1,
                    'percent_overlap': 80,
                    'n_mfcc' : 64,
                    'delta': False,
                    'doubledelta': False,
            
                    'label_type': 'normal' ,#'normal'
                    'type': None
                    }

cnngunshotParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':128,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.1,
                    'percent_overlap': 80,
                    'n_mfcc' : 64,
                    'delta': False,
                    'doubledelta': False,
            
                    'label_type': 'normal' ,#'normal'
                    'type': None }


cnnParams = { 'babycry' : cnnbabycryParams,
              'glassbreak' : cnnglassbreakParams,
              'gunshot' : cnngunshotParams}


rnnbabycryParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.120,
                    'percent_overlap': 50,
                    'n_mfcc' :64 ,
                    'delta':False,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': False}

rnnglassbreakParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.120,
                    'percent_overlap': 50,
                    'n_mfcc' :64 ,
                    'delta':False,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': False}

rnngunshotParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':64,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.120,
                    'percent_overlap': 50,
                    'n_mfcc' :64,
                    'delta':True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}

rnnParams = { 'babycry' : rnnbabycryParams,
              'glassbreak' : rnnglassbreakParams,
              'gunshot' : rnngunshotParams}


source_babycryParams = {'sr':44100,
                    'n_fft':8192,
                    'n_mels':128,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.1,
                    'percent_overlap': 80,
                    'n_mfcc' : 64,
                    'delta': False,
                    'doubledelta': False,
            
                    'label_type': 'normal' ,#'normal'
                    'type': None}  #must be same as for mixture data. Can't change.
                    
source_glassbreakParams = {'sr':44100,
                    'n_fft':2048,
                    'n_mels':40,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.04,
                    'percent_overlap': 50,
                    'n_mfcc' : 20,
                    'delta': True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}
                    
source_gunshotParams = {'sr':44100,
                    'n_fft':4096,
                    'n_mels':128,
                    'fmin': 0,
                    'fmax': 22050,
                    'win_length' : 0.08,
                    'percent_overlap': 50,
                    'n_mfcc' : 120,
                    'delta': True,
                    'mfcc0': False,
                    'label_type': 'normal' ,
                    'doubledelta': True}

source_dataParams = {'babycry': source_babycryParams,
                    'glassbreak': source_glassbreakParams,
                    'gunshot': source_gunshotParams }
                    

d1 = {'dnn': dnnParams, 
      'cnn' : cnnParams,
      'rnn' : rnnParams,
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



#featureParams = loadfeatureParams(yamlfile)

#print(featureParams)


dnnbabycryParams = { 'num_layers': 3,
              'batch_norm': [True,True, True],
              'num_units':[100,100,100],
              'Dropout': [0.2,0.2,0.3],
               'lr': 0.0020,
               'epochs': 5,
               'batch_size': 256,
               'l2reg': 1e-7,
               'input_dimension':531,
               'eval_predict': True,
               'classify_threshold':0.60,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-3,
               'loss' :'binary_crossentropy',
               'context':3,
               'label_group_threshold':0.25,
               'medfilt_kernel':3,
               'override_epochs':False,
               'load_model': 'load', #'load'
               'model_path': 'dnn_02_14_PM_August_03_2017_babycry_04_46_PM_August_05_2017_ModelFile.h5', #path to model file
               'train_model': True, 
               'do_prediction': True,
               'save_model': True,
               'directory' :  'dnn_02_14_PM_August_03_2017', #'dnn_04_49_PM_July_17_2017',#dnn_05_06_PM_July_27_2017',
               'class_weights' : {0:10, 1:1},
                'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }



dnnglassbreakParams =  { 'num_layers': 3,
              'batch_norm': [True,True,True],
              'num_units':[100,100,100],
              'Dropout': [0.2,0.2,0.2],
               'lr': 0.0033,
               'epochs':3,
               'batch_size': 256,
               'l2reg': 1e-7,
               'eval_predict': True,
               'input_dimension':531,
               'classify_threshold':0.63,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-5,
               'override_epochs':False,
               'loss' :'binary_crossentropy',
               'context':3,
               'load_model': 'load', #'load'
               'model_path': 'dnn_10_06_PM_August_02_2017_glassbreak_09_20_AM_August_03_2017_ModelFile.h5', #path to model file
               'train_model': False, 
               'do_prediction': True,
               'save_model': False,
               'directory' : 'dnn_10_06_PM_August_02_2017',
               'label_group_threshold':0.3,
               'medfilt_kernel':5,
               'class_weights' : {0:10, 1:1},
               'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }


dnngunshotParams =  { 'num_layers': 3,
              'batch_norm': [True,True,True],
              'num_units':[100, 50,50,],
              'Dropout': [0.4,0.4,0.2],
               'lr': 0.003,
               'epochs': 7,
               'batch_size': 128,
               'l2reg': 5e-5,
               'input_dimension':295,
               'classify_threshold':0.45,
               'eval_predict': True,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-3,
               'override_epochs':False,
               'loss' :'binary_crossentropy',
               'context':5,
               'load_model': 'new', #'load'
               'model_path': None, #path to model file
               'train_model': True, 
               'do_prediction': True,
               'save_model': True,
               'directory' : 'dnn_11_18_AM_August_02_2017',
               'label_group_threshold':0.125,
               'medfilt_kernel':3,
               'class_weights' : {0:10, 1:1},
                'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }










cnnbabycryParams = {'BatchNorm': True, 
                    'l2reg':9e-5,
                    'xdim': 16,
                    'ydim' :  64,
                    'num_channels': 1,
                   'optimizer':'sgd',
                   'classify_threshold':0.5,
                   'decay' : 5e-3,
                   'lr': 0.005,
                   'eval_predict': True,
                   'epochs':2,
                   'context':16,
                   'override_epochs':False,
                   'medfilt_kernel':7,
                   'directory' : 'cnn_12_01_PM_August_05_2017', #'cnn_10_09_PM_July_30_2017',  # 'cnn_03_34_PM_July_30_2017',
                   'batch_size': 64,
                   'load_model': 'new', #'load'
                   'model_path': None, #'cnn_12_01_PM_August_05_2017_babycry_01_34_PM_August_05_2017_ModelFile.h5', #'cnn_10_09_PM_July_30_2017_babycry_ModelFile.h5', #path to model file
                   'train_model': True, 
                   'do_prediction': True,
                   'save_model': True,
                   'loss' :'binary_crossentropy',
                   'label_group_threshold': 0.125,
                   'class_weights' : {0:10, 1:1},
                   'weight_init':  'glorot_uniform'  #'glorot_normal'
              }


cnnglassbreakParams = {'BatchNorm': True, 
                    'l2reg':8e-5,
                    'xdim': 16,
                    'ydim' : 128,
                    'num_channels': 1,
                   'optimizer':'sgd',
                   'classify_threshold':0.7,
                   'decay' : 5e-4,
                   'lr': 0.0055,
                   'eval_predict': True,
                   'epochs':3 ,
                   'context':16,
                   'medfilt_kernel':9,
                   'override_epochs':False,
                   'directory' : 'cnn_05_05_PM_August_02_2017', #'cnn_01_47_PM_August_01_2017',  # 'cnn_03_34_PM_July_30_2017',
                   'batch_size': 64,
                   'load_model': 'new', #'load'
                   'model_path': None, #path to model file
                   'train_model': True, 
                   'do_prediction': True,
                   'save_model': True,
                   'loss' :'binary_crossentropy',
                   'label_group_threshold': 0.125,
                   'class_weights' : {0:10, 1:1},
                   'weight_init':  'glorot_uniform'  #'glorot_normal'
              }


cnngunshotParams = {'BatchNorm': True, 
                    'l2reg':8e-5,
                    'xdim': 16,
                    'ydim' : 128,
                    'num_channels': 1,
                   'optimizer':'sgd',
                   'classify_threshold':0.6,
                   'decay' : 5e-3,
                   'eval_predict': True,
                   'lr': 0.005,
                   'epochs':3,
                   'context':16,
                   'medfilt_kernel':7,
                   'override_epochs':False,
                   'directory' :'cnn_11_25_AM_August_01_2017',  # 'cnn_03_34_PM_July_30_2017',
                   'batch_size': 64,
                   'load_model': 'load', #'load'
                   'model_path': 'cnn_11_25_AM_August_01_2017_gunshot_01_19_PM_August_02_2017_ModelFile.h5', #'cnn_11_25_AM_August_01_2017_gunshot_01_19_PM_August_02_2017_ModelFile.h5', #path to model file
                   'train_model': False, 
                   'do_prediction': True,
                   'save_model': False,
                   'loss' :'binary_crossentropy',
                   'label_group_threshold': 0.125,
                   'class_weights' : {0:10, 1:1},
                   'weight_init':  'glorot_uniform'  #'glorot_normal'
              }





rnnbabycryParams = { 
              
               'lr': 0.0045,#0.0045,
               'epochs': 4,
               'batch_size':8 ,
               'l2reg': 1e-4,
               'input_dimension':189, #189,
               'classify_threshold':0.55,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-2,#5e-3,
               'loss' :'binary_crossentropy',
               'context':3,
               'eval_predict': True,
               'override_epochs':False,
               'label_group_threshold':0.20,
               'medfilt_kernel':5,
               'directory' : None, #'rnn_01_57_PM_July_31_2017', #'rnn_01_57_PM_July_31_2017', # 'rnn_11_49_AM_July_31_2017',
               'class_weights' : {0:10, 1:1},
               'load_model': 'new', #'load'
               'model_path': None, #'rnn_01_57_PM_July_31_2017_babycry_04_14_PM_August_03_2017_ModelFile.h5',# 'rnn_01_57_PM_July_31_2017_babycry_03_04_PM_August_03_2017_ModelFile.h5', #path to model file
               'train_model': True, 
               'do_prediction': True,
               'save_model': True,
                'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }



rnnglassbreakParams =  { 
               'lr': 0.0045,
               'epochs': 12,
               'batch_size':8 ,
               'l2reg': 1e-5,
               'input_dimension':189,
               'classify_threshold':0.75,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-3,
               'eval_predict': True,
               'loss' :'binary_crossentropy',
               'context':3,
               'override_epochs':False,
               'label_group_threshold':0.4,
               'medfilt_kernel':9,
               'directory' :'rnn_09_29_AM_August_03_2017', # 'rnn_11_49_AM_July_31_2017',
               'class_weights' : {0:10, 1:1},
               'load_model': 'new', #'load'
               'model_path': None, #'rnn_09_29_AM_August_03_2017_glassbreak_11_45_AM_August_03_2017_ModelFile.h5', #'rnn_09_29_AM_August_03_2017_glassbreak_10_01_AM_August_03_2017_ModelFile.h5', #path to model file
               'train_model': True, 
               'do_prediction': True,
               'save_model': True,
                'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }


rnngunshotParams =  {
                'lr': 0.0052,
               'epochs': 5,
               'batch_size':8 ,
               'l2reg': 5e-5,
               'input_dimension':567,
               'classify_threshold':0.5,
               'optimizer':'sgd',   #'adagrad'
               'decay' : 5e-3,
               'eval_predict': True,
               'loss' :'binary_crossentropy',
               'context':3,
               'override_epochs':False,
               'label_group_threshold':0.125,
               'medfilt_kernel':3,
               'directory' : 'rnn_10_10_PM_August_04_2017', #'rnn_11_12_AM_August_02_2017', # 'rnn_11_49_AM_July_31_2017',
               'class_weights' : {0:10, 1:1},
               'load_model': 'load', #'load'
               'model_path': 'rnn_10_10_PM_August_04_2017_gunshot_11_47_PM_August_04_2017_ModelFile.h5',#'rnn_11_12_AM_August_02_2017_gunshot_01_41_PM_August_02_2017_ModelFile.h5', #path to model file
               'train_model': True, 
               'do_prediction': True,
               'save_model': True,
                'weight_init':  'glorot_uniform'  #'glorot_normal'
                # 'mean_squared_error'
                }





cnnParams = {'babycry':cnnbabycryParams,
             'glassbreak': cnnglassbreakParams,
             'gunshot': cnngunshotParams }

rnnParams = {'babycry':rnnbabycryParams,
             'glassbreak': rnnglassbreakParams,
             'gunshot': rnngunshotParams }

dnnParams = {'babycry': dnnbabycryParams,
             'glassbreak': dnnglassbreakParams,
             'gunshot': dnngunshotParams }

modelParams = { 'dnn': dnnParams, 
                'cnn' : cnnParams,
                'rnn' : rnnParams,
                'tuning_mode': False,
                'postprocess_tune':False,  #'exit' to save predictions and exit
                'tuning_countmax':50, #if tuning enable, how many trial runs to run.
                'min_delta' :0.001,
                'patience':4,
                'class_weights_enable': False,
                'pp_tuning_countmax':75,
                'simulate_single_run':True,
                'extraction_mode':None,     #'ask' to get user input regarding which paramHash to consider
                'ensemble_dnn_weight': 0.5,
                'ensemble_type' : 'average_before',   #write code for average_after

                'train_feature_norm': False,
                'test_feature_norm': False,
                'shuffle_data': False,

                'train_feature_ext': False,
                'train_label_ext': False,
                'test_feature_ext': False,
                'test_label_ext': False,
                'newTest_feature_ext': False,
                'newTest_label_ext': False,
                'eval_feature_ext': False,
                
              
                'dump_cnn_train_data': True,
                'dump_cnn_test_data': False,
                'dump_cnn_newTest_data': False,
                'dump_cnn_eval_data': False,

                'dump_dnn_train_data':False,
                'dump_dnn_test_data':False,
                'dump_dnn_newTest_data': False,
                'dump_dnn_eval_data': False,

                'dump_rnn_train_data':False,
                'dump_rnn_test_data':False,
                'dump_rnn_newTest_data':False,
                'dump_rnn_eval_data': False,

                'mycomp':True,

                }



dnnBatchParams = {  'generate_train_total': 82,
                    'generate_train_batchsize':9724,
                    'generate_val_total': 85,
                    'generate_val_batchsize': 8800,
                    
                    
                    'predict_iterations':  85,
                    'predict_batchsize': 8800,
                    'newTest_iterations': 85 ,
                    'newTest_batchsize': 8800 ,


                    'eval_iterations':  85,
                    'eval_batchsize': 8800
}


cnnBatchParams = {  'generate_train_total':18, #54,#246, #54,  #1463671-->1463670
                    'generate_train_batchsize':40262, #24600, #7500, #24600,
                    'generate_val_total': 18,
                    'generate_val_batchsize': 40262,
                   
                   
                    'predict_iterations': 62,
                    'predict_batchsize': 11808,
                    'newTest_iterations': 41,
                    'newTest_batchsize': 18000,


                     'eval_iterations':  41,
                    'eval_batchsize': 18000
}

rnnBatchParams = {  'generate_train_total':5 ,
                    'generate_train_batchsize':100,
                    'generate_val_total': 5,
                    'generate_val_batchsize': 100,
                  
                   
                    'predict_iterations':  5,
                    'predict_batchsize': 100,
                    'newTest_iterations': 5 ,
                    'newTest_batchsize': 100,

                     'eval_iterations':  5,
                    'eval_batchsize': 100
}



batch_training_params = {'dnn': dnnBatchParams,
                          'cnn': cnnBatchParams,
                          'rnn': rnnBatchParams
                          
                          }


def createBatchTrainingParams():
  yamlfile = DumpfeatureParams(batch_training_params,
                              'batchTrainingParams.yaml')

  

def createParamsFile():
  yamlfile = DumpfeatureParams(modelParams,'modelParams.yaml')

  file1 = 'featureParams.yaml'
  yamlfile = DumpfeatureParams(d1,file1)

#modelParamsReloaded = loadfeatureParams(yamlfile)

#print(modelParamsReloaded)

