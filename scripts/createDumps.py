import os 
import shutil
import pandas as pd

data_path = '/home/lizano/Documents/SAC/data/raw/cnn/unclassified_raw_data'
os.system('rm -rf '+os.path.join(data_path,'*'))
destination_path = os.path.join(data_path,'full')
os.mkdir(destination_path)
data = pd.DataFrame(columns=['Image Names','C6_avg','Psi6'])

#voltages = ['V1', 'V2', 'V3', 'V4']
#steps = ['5s','10s','30s']

voltages = ['V1','V2','V3','V4']
steps = ['5s','10s','30s']

path = '/home/lizano/Documents/SAC/data/raw/snn'
    
for V in voltages:
    V_path = os.path.join(path,V)
    if os.path.exists(V_path):
        for S in steps:
            S_path = os.path.join(V_path,S)
            if os.path.exists(S_path):
                for T in os.listdir(S_path):
                    new_path = os.path.join(S_path,T)
                    local_data = pd.read_csv(os.path.join(new_path,V+'-'+S+'-'+T+'.csv'))
                    for i in range(len(local_data)):
                        local_file = os.path.join(new_path,'plots') 
                        name = V+'-'+T+'-'+str(i)+'step'+S+'.png'
                        c6 = local_data['C6_avg'].iloc[i]
                        psi6 = local_data['psi6'].iloc[i]
                        source = os.path.join(local_file,name)
                        destination = os.path.join(destination_path,name)
                        row = {'Image Names':name,'C6_avg':c6,'Psi6':psi6}
                        data = data.append(row,ignore_index=True)
                        try:
                            shutil.copy(source, destination)
                        except:
                            print("Error occurred while copying file: ", name)

print(data.head())
data.to_csv(os.path.join(data_path,'full_dataset_order_parameters.csv'))
                    