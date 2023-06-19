import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.operators.bash_operator import BashOperator

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from airflow.models import DagRun
from airflow.utils.state import State
import mlflow
import mlflow.tensorflow
from flask import Flask, request
import hmac
import hashlib
import datetime
from airflow.utils import timezone
from flask import Flask, request, jsonify
import numpy as np
EXP_NAME = 'cifar10_REDO_Github'
RUN_NAME = 'Dummy1'

# app = Flask(__name__)
# # Define the webhook handler function
# def handle_webhook(dag_run_payload):
#     dag_run = DagRun.find(dag_id=EXP_NAME, state='running')
#     if dag_run:
#         print(f"DAG run for '{EXP_NAME}' is already in progress.")
#     else:
#         run_id = f"{EXP_NAME}_{datetime.datetime.now().isoformat()}"
#         run_conf = {'payload': dag_run_payload}
#         dag.create_dagrun(
#             run_id=run_id,
#             state=State.RUNNING,
#             execution_date=timezone.utcnow(),
#             conf=run_conf,
#             external_trigger=True,
#         )
#         print(f"DAG run '{run_id}' triggered for '{EXP_NAME}'.")

app = Flask(__name__)


# Define the webhook handler function
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    event = request.headers.get('X-GitHub-Event')
    if event == 'push':
        payload = request.json
        # Extract relevant information from the payload, e.g., repository, commit, file changes, etc.
        # Trigger the Airflow DAG based on the extracted information using the `trigger_dag` method or create a new DAG run
        trigger_airflow_dag(payload)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'ignored'})


def trigger_airflow_dag(payload):
    dag_run = DagRun.find(dag_id=EXP_NAME, state='running')
    if dag_run:
        print(f"DAG run for '{EXP_NAME}' is already in progress.")
    else:
        run_id = f"{EXP_NAME}_{datetime.datetime.now().isoformat()}"
        run_conf = {'payload': payload}
        dag.create_dagrun(
            run_id=run_id,
            state=State.RUNNING,
            execution_date=timezone.utcnow(),
            conf=run_conf,
            external_trigger=True,
        )
        print(f"DAG run '{run_id}' triggered for '{EXP_NAME}'.")

def setup_mlflow():
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    try:
        mlflow.create_experiment(EXP_NAME)
        print(f"Experiment '{EXP_NAME}' created.")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Experiment '{EXP_NAME}' already exists.")
        else:
            raise


def start_mlflow_run(name='MainConfig'):
    mlflow.set_experiment(EXP_NAME)
    mlflow.start_run(run_name=name, nested=True)


def end_mlflow():
    mlflow.end_run()


def download_dataset(**context):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    os.makedirs('data', exist_ok=True)
    x_train_file = "data/x_train.npy"
    y_train_file = "data/y_train.npy"

    x_test_file = "data/x_test.npy"
    y_test_file = "data/y_test.npy"

    np.save(x_train_file, x_train)
    np.save(y_train_file, y_train)

    np.save(x_test_file, x_test)
    np.save(y_test_file, y_test)

    context['ti'].xcom_push(key='x_train_file', value=x_train_file)
    context['ti'].xcom_push(key='y_train_file', value=y_train_file)

    context['ti'].xcom_push(key='x_test_file', value=x_test_file)
    context['ti'].xcom_push(key='y_test_file', value=y_test_file)


def normalize_dataset(**context):
    x_train_file = context['ti'].xcom_pull(key='x_train_file')
    x_test_file = context['ti'].xcom_pull(key='x_test_file')

    x_train = np.load(x_train_file)
    x_test = np.load(x_test_file)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train_normalized_file = "data/x_train_normalized.npy"
    x_test_normalized_file = "data/x_test_normalized.npy"

    np.save(x_train_normalized_file, x_train)
    np.save(x_test_normalized_file, x_test)

    context['ti'].xcom_push(key='x_train_normalized_file', value=x_train_normalized_file)
    context['ti'].xcom_push(key='x_test_normalized_file', value=x_test_normalized_file)


def build_classifier(**context):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    os.makedirs('model', exist_ok=True)
    model.save("model/saved_model.h5")
    context['ti'].xcom_push(key='model_path', value="model/saved_model.h5")


def train_model(**context):
    start_mlflow_run(name=RUN_NAME)
    x_train_normalized_file = context['ti'].xcom_pull(key='x_train_normalized_file')
    y_train_file = context['ti'].xcom_pull(key='y_train_file')
    model_path = context['ti'].xcom_pull(key='model_path')

    x_train_normalized = np.load(x_train_normalized_file)
    y_train = np.load(y_train_file)

    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train_normalized, y_train, epochs=5)

    mlflow.log_metric('training_loss', history.history['loss'][-1])
    mlflow.log_metric('training_accuracy', history.history['accuracy'][-1])
    context['ti'].xcom_push(key='best_model_path', value=model_path)


def evaluate_model(**context):
    x_train_normalized_file = context['ti'].xcom_pull(key='x_test_normalized_file')
    y_train_file = context['ti'].xcom_pull(key='y_test_file')
    model_path = context['ti'].xcom_pull(key='best_model_path')
    x_train_normalized = np.load(x_train_normalized_file)
    y_train = np.load(y_train_file)

    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    loss, accuracy = model.evaluate(x_train_normalized, y_train)
    context['ti'].xcom_push(key='model_accuracy', value=accuracy)

    mlflow.log_metric('valid loss', loss)
    mlflow.log_metric('valid accuracy', accuracy)

    end_mlflow()



default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2023, 6, 19),
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'provide_context': True,
}


with DAG(EXP_NAME,
         default_args=default_args,
         schedule_interval='@once') as dag:

    webhook_sensor = FileSensor(
        task_id='webhook_sensor',
        poke_interval=10,
        fs_conn_id='github_webhook',
        filepath='webhook_trigger',
        dag=dag
    )


    mlflow_start_task = PythonOperator(
        task_id='starting_MLFlow_Server',
        python_callable=setup_mlflow,
        provide_context=True
    )

    download_task = PythonOperator(
        task_id='download_dataset',
        python_callable=download_dataset,
        provide_context=True
    )

    normalize_task = PythonOperator(
        task_id='normalize_dataset',
        python_callable=normalize_dataset,
        provide_context=True
    )

    build_classifier_task = PythonOperator(
        task_id='build_classifier',
        python_callable=build_classifier,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True
    )

    cleanup_task = BashOperator(
        task_id='cleanup_files',
        bash_command='rm -f *.npy',
    )

    end_task = DummyOperator(task_id='end')

    webhook_sensor >> mlflow_start_task
    mlflow_start_task >> download_task
    download_task >> normalize_task

    normalize_task >> build_classifier_task
    build_classifier_task >> train_model_task
    train_model_task >> evaluate_model_task
    evaluate_model_task >> cleanup_task >> end_task


# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
