import h5py




    
def ff():
    with h5py.File("/hdd4/cnn_12_01_PM_August_05_2017_babycry_newTest_features.h5",'r') as f:
        d = f['features']
        print("newTest: ", d.shape)


    with h5py.File("/hdd4/cnn_12_01_PM_August_05_2017_babycry_devtest_features.h5",'r') as f:
        d = f['features']
        print("devTest: ", d.shape)


    with h5py.File("/hdd4/cnn_12_01_PM_August_05_2017_babycry_devtrain_features.h5",'r') as f:
        d = f['features']
        print("devtrain: ", d.shape)



ff()