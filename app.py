from flask  import Flask, render_template, request

import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,MinMaxScaler


app = Flask(__name__,template_folder='template/')

with open('machine_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/',methods=['GET']) 
def Home():
    return render_template('index.html',pred=' ')

@app.route("/", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        # print(request.form)
        product_id = request.form['id']
        type_var = request.form['Type']
        rot_speed = int(request.form['rot_speed'])
        air_temp = float(request.form['Air_temp'])
        process_temp = float(request.form['Process_temp'])
        torque = float(request.form['Torque'])
        tool_wear = int(request.form['tool_wear'])
        twf = int(request.form['TWF'])
        hdf = int(request.form['HDF'])
        pwf = int(request.form['PWF'])
        osf = int(request.form['OSF'])
        rnf = int(request.form['RNF'])

        input_data = {
            'product_id': [product_id],
            'Type': [type_var],
            'air_temp': [air_temp],
            'process_temp': [process_temp],
            'rot_speed': [rot_speed],
            'torque': [torque],
            'tool_wear_min': [tool_wear],
            'TWF': [twf],
            'HDF': [hdf],
            'PWF': [pwf],
            'OSF': [osf],
            'RNF': [rnf]
        }

        input_df = pd.DataFrame(input_data)

        LE = LabelEncoder()
        sc = MinMaxScaler()

        input_df['Type'] = LE.fit_transform(input_df['Type'])
        input_df.iloc[:,2:7] = sc.fit_transform(input_df.iloc[:,2:7])


        prediction = loaded_model.predict(input_df.iloc[:,:])

        
        if prediction[0] == 0:
            return render_template('index.html',pred="Machine is working fine")
        else:
            return render_template('index.html',pred="Machine failure!")
    else:
        return render_template('index.html', pred=' ')

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)

