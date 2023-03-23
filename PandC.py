import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import MinMaxScaler

class CampusPlacementPrediction():
    '''
        A class to represent a prediction of campus placement.

        ...

        Attributes
        ----------
        model file : file
                    predictive model saved after the experiment
                In this case it is the 'Neural Network Regression' model

        Methods
            -------
            load_clean_data(data_file):

            # take a data file (*.txt) and preprocess it
                Import the text file, it processes , clean and standardize the file required for prediction
            Parameters

        predicted_vallue():

                Processed data will be predicted.

        predicted_outputs():

             Processed data will be predicted and concated with Original Value.
        pass'''

    def __init__(self, model_files):

        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            model file : file
                predictive model saved after the experiment
            In this case it is the 'Neural Network Regression' model

        """
        # read the 'model' files which were saved
        with open(model_files, 'rb') as model_file:
            self.classification = pickle.load(model_file)

    def load_clean_data(self, df):
        # take a dataframe file
        """
            Import the csv file and it process and clean and standardize the file required for prediction
        Parameters
        ----------
        data_file : in .txt format

        Returns
        -------
        cleaned and processed file required for prediction

        '''
        """

        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()

        ## One Hot encoding
        df['hsc_s_Science']= df['hsc_type'].apply(lambda x: 1 if x == 'hsc_s_Science' else 0)
        df['hsc_s_Commerce'] = df['hsc_type'].apply(lambda x: 1 if x == 'hsc_s_Commerce' else 0)
        df['hsc_s_Arts'] = df['hsc_type'].apply(lambda x: 1 if x == 'hsc_s_Arts' else 0)

        ## Droping the unwanted Column
        df.drop(['hsc_type'], axis=1, inplace=True)

        # re-order the columns in df
        Column_names = ['gender', 'spec', 'work', 'ssc', 'hsc', 'dsc', 'mba', 'etet', 'hsc_s_Science', 'hsc_s_Commerce',
                        'hsc_s_Arts']

        df = df[Column_names]

        self.preprocessed_data = df.copy()

        sc = MinMaxScaler()

        self.data = sc.fit_transform(df)

    def predicted_vallue(self):
        """
            Processed data will be predicted.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            pred = self.classification.predict(self.data)[:, 1]
            return pred

    # predict the outputs and
    # add columns with these values at the end of the new data

    def predicted_outputs(self):
        """
            Processed data will be predicted and concated with Original Value.
        ----------

        Returns
        -------
        Predicted values
        """
        if (self.data is not None):
            self.prediction = self.classification.predict(self.data)
            self.preprocessed_data['Prediction'] = self.prediction
            return self.preprocessed_data