from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from TaxiFareModel.utils import compute_rmse 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient
import joblib



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = '[GLO][Global][rreinhaus]TaxiFareModel 1'

    def set_pipeline(self, model):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])], remainder="drop")
        pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('model', model())])

        return pipe

    def run(self):
        """set and train the pipeline"""
        pipe = self.set_pipeline(model)
        pipe.fit(X_train, y_train)
        return pipe

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipe = self.run()
        y_pred = pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri('https://mlflow.lewagon.co/')
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.run(), 'model.joblib')



if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    # build pipeline
    tr_pipeline = Trainer(X,y)
    for model in [LinearRegression, RandomForestRegressor]:
        pipeline = tr_pipeline.set_pipeline(model)
        trainer = Trainer(X_train, y_train)
        # train
        trainer.run()
        # evaluate the pipeline
        rmse = trainer.evaluate(X_val, y_val)
        print(rmse)
        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
        # save the model
        trainer.save_model()
        # running the model
        client = trainer.mlflow_client
        run = trainer.mlflow_run
        trainer.mlflow_log_metric("rmse", rmse)
        trainer.mlflow_log_param("model", model)
        trainer.mlflow_log_param("student_name", 'rreinhaus')