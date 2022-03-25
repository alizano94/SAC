import os
import shutil
import pandas as pd

def copy_from_csv(data,origin,destination):
    '''
    Function that copy elements in data from 
    from on directory to another.
    args:
        -data: datframe containing dataset info
        -origin: directory from which copy files
        -destination: directory to copy file to
    return:
    '''
    n = 0
    for file in data['Image Names'].to_list():
        source = os.path.join(origin,'full',file)
        print(source)
        try:
            shutil.copy(source, destination)
        except:
            print("Error occurred while copying file.")
            print(file)
            n += 1
    print('Number of errors: ', n)

data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
data_file = 'full_dataset_order_parameters.csv'

train_data_path = os.path.join(data_path,'train')
try:
    shutil.rmtree(train_data_path)
except:
    os.mkdir(train_data_path)
test_data_path = os.path.join(data_path,'test')
try:
    shutil.rmtree(test_data_path)
except:
    os.mkdir(test_data_path)
validation_data_path = os.path.join(data_path,'validation')
try:
    shutil.rmtree(validation_data_path)
except:
    os.mkdir(validation_data_path)

training_size = 4000
testing_size = int(0.25*training_size)
validation_size = int(0.25*training_size)
sampling_size = training_size+testing_size+validation_size

data_df = pd.read_csv(os.path.join(data_path,data_file), index_col=0)

sample = data_df.sample(sampling_size)

training_df = sample.iloc[0:training_size]
testing_df = sample.iloc[training_size:training_size+testing_size]
validation_df = sample.iloc[training_size+testing_size:training_size+testing_size+validation_size]

try:
    testing_df.to_csv(os.path.join(data_path,'testing_set_order_parameters.csv'))
    copy_from_csv(testing_df,data_path,test_data_path)
    training_df.to_csv(os.path.join(data_path,'training_set_order_parameters.csv'))
    copy_from_csv(training_df,data_path,train_data_path)
    validation_df.to_csv(os.path.join(data_path,'validation_set_order_parameters.csv'))
    copy_from_csv(validation_df,data_path,validation_data_path)
    print('Randomlly sampled dataset from:',data_path)
    print('Training data set size:',len(training_df))
    print('Testing data set size:',len(testing_df))
    print('Validation data set size:',len(validation_df))
except:
    print('Try again!')