import os 
import shutil
import pandas as pd

destination_path = '/home/lizano/Documents/SAC/data/raw/cnn/dump/0'
data = pd.DataFrame(columns=['name','C6_avg','Psi6'])

#voltages = ['V1', 'V2', 'V3', 'V4']
#steps = ['5s','10s','30s']

voltages = ['V1','V2','V3','V4']
steps = ['5s','10s','30s']

for V in voltages:
    path = '/home/lizano/Documents/SAC/data/raw/snn'
    path = os.path.join(path,V)
    if os.path.exists(path):
        for S in steps:
            path = os.path.join(path,S)
            if os.path.exists(path):
                for T in os.listdir(path):
                    new_path = os.path.join(path,T)
                    local_data = pd.read_csv(os.path.join(new_path,V+'-'+S+'-'+T+'.csv'))
                    for i in range(len(local_data)):
                        local_file = os.path.join(new_path,'plots') 
                        name = V+'-'+T+'-'+str(i)+'step'+S+'.png'
                        c6 = local_data['C6_avg'].iloc[i]
                        psi6 = local_data['psi6'].iloc[i]
                        source = os.path.join(local_file,name)
                        destination = os.path.join(destination_path,name)
                        row = {'name':name,'C6_avg':c6,'Psi6':psi6}
                        data = data.append(row,ignore_index=True)
                        try:
                            shutil.copy(source, destination)
                            print("File copied successfully.")
                        except:
                            print("Error occurred while copying file.")

print(data.head())
data.to_csv(os.path.join(destination_path,'data_parameters.csv'))
                    