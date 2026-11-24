from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import timedelta


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("nova_training", default_args=default_args)


log_datetime_start_task = BashOperator(
    task_id="log_datetime_start", bash_command="date", dag=dag
)

preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --preprocess",
    image="nova:latest",
    network_mode="infra_default",
    task_id="preprocessing",
    mounts=[
        Mount(
            source="data", target="/nova/data"
        )  # note: Normally we do not do this via local files. I would use a cloud storage system like AWS S3 or Azure Blob Storage.
    ],
    dag=dag,
)

feat_eng_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --feat-eng",
    image="nova:latest",
    network_mode="infra_default",  # note: In order services to communicate each other, we add them to the same network. You do not see in docker-compose.yaml, b/c by default it puts all the services under one network called "infra_default".
    task_id="feature_engineering",
    mounts=[Mount(source="data", target="/nova/data")],
    dag=dag,
)

completely_unnecessary_task_to_display_parallelism = BashOperator(
    task_id="Chatter",
    bash_command="for i in {1..10}; do date +'%Y-%m-%d %H:%M:%S'; sleep 5; done",
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --training",
    image="nova:latest",
    network_mode="infra_default",
    task_id="training",
    mounts=[Mount(source="data", target="/nova/data")],
    dag=dag,
)

log_datetime_end_task = BashOperator(
    task_id="log_datetime_end", bash_command="date", dag=dag
)


(
    log_datetime_start_task
    >> preprocessing_task
    >> [feat_eng_task, completely_unnecessary_task_to_display_parallelism]
    >> model_training_task
    >> log_datetime_end_task
)
