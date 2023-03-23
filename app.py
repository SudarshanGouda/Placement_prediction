from flask import Flask, render_template, request
from PandC import *

# unplickling the model

file = open('finalized_model.pkl', 'rb')
rf = pickle.load(file)
file.close()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        gender = int(mydict['gender'])
        spec = int(mydict['spec'])
        work = int(mydict['work'])
        ssc = float(mydict['ssc'])
        hsc = float(mydict['hsc'])
        dsc = float(mydict['dsc'])
        mba = float(mydict['mba'])
        etet = float(mydict['etet'])
        hsc_type = str(mydict['hsc_type'])

        inputfeatures = [[gender, spec, work, ssc, hsc, dsc, mba, etet, hsc_type]]

        df = pd.DataFrame(inputfeatures,
                          columns=['gender', 'spec', 'work', 'ssc', 'hsc', 'dsc', 'mba', 'etet', 'hsc_type'])
        model = CampusPlacementPrediction('./finalized_model.pkl')
        model.load_clean_data(df)
        presicted_df = model.predicted_outputs()
        presicted_df.to_csv('Final_prediction.csv')
        result=int(presicted_df['Prediction'])

        if result == 1:
            string = 'You Have got High Chance of Getting Placed'
            return render_template('final.html', string=string)
        else:
            string = 'You Have got Low Chance of Getting Placed'
            return render_template('final.html', string=string)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)