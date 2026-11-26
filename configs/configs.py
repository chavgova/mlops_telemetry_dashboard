from dataclasses import dataclass, field


@dataclass(init=False, frozen=True)
class PathsConfig:
    raw_data_path: str = "data/trx/"  # Raw upstream data.
    preprocessing_data_path: str = "data/intermediate/preprocessing/"
    features_data_path: str = "data/intermediate/features/"

    mean_amount_all_categories_file_name = "_mean_amount_all_categories.pkl"
    std_amount_all_categories_file_name = "std_amount_all_categories.pkl"
    mean_amount_per_category_file_name = "mean_amount_per_category.pkl"
    std_amount_per_category_file_name = "std_amount_per_category.pkl"
    indexer_per_column_file_name_prefix = "indexer_per_column_"
    encoder_per_column_file_name_prefix = "encoder_per_column_"
    column_path_map_file_name = "column_path_map.pkl"

    telemetry_training_data_path: str = "data/telemetry/data_dist.json"
    telemetry_live_data_path: str = "data/telemetry/live_data_dist.json"


@dataclass(frozen=True)
class _RunConfig:
    """
    This class should not be used directly without instantiation. Always use the run_config in this module.
    """

    app_name: str = "nova"
    spark_master_url: str = "local[2]"
    # The fraction of the entire raw data to be selected at random.
    # For the sake of speed in development, set this to smaller number. In production, always expected to be
    # set to 1.0.
    sample_rate: float = 1.0
    num_folds: int = 5
    threshold_in_eur_to_consider_large_transactions: float = 123.45
    mlflow_tracking_uri: str = "http://mlflow:8080"  # "http://localhost:8080"
    experiment_name: str = "Alpha"
    run_name: str = "Run_234"
    categorical_cols: list = field(default_factory=lambda: ["category", "payment_type"])


@dataclass(frozen=True)
class TelemetryConfig:
    # The number of instances that should be in the live distribution.
    # As we will need a distribution that represents the "recent" status, we will need to form a distribution from
    # the "latest" data that the application has received. For this system, we define "latest" as the num_instances_for_live_dist
    # instances that the app received.
    # Therefore, we take the latest num_instances_for_live_dist instances as the live distribution for the
    # calculation of PSI.
    #
    # Example:
    #       num_instances_for_live_dist = 2
    #
    #   Call Time,  row id,     value
    #    10:30        1           0.1
    #    10:31        2           0.2
    #    11:32        3           0.3
    #
    # Live/"latest" distribution: [0.2, 0.3]
    #
    num_instances_for_live_dist: int = 100
    epsilon: float = 1e-6
    live_window_seconds: int = 300
    push_gateway_uri: str = (
        "http://prometheus_push_gateway:9091"  # "http://localhost:9091"
    )


@dataclass(
    init=False,
)
class ModelConfig:
    model_name: str = "alpha"
    max_depth: int = 100
    class_weight: str = "balanced"


run_config = _RunConfig()
