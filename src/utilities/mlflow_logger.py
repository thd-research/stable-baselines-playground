import inspect
from datetime import datetime
import mlflow
import numpy as np
import os
import sys
from typing import Dict, Any, Tuple, Union

from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from dataclasses import is_dataclass, fields


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def get_ml_logger(debug=False):
    output_formats = [MLflowOutputFormat()]
    if debug:
        print("Debug is on")
        output_formats += [HumanOutputFormat(sys.stdout)]

    loggers = Logger(
        folder=None,
        output_formats=output_formats,
    )
    return loggers


def mlflow_monotoring(subfix=""):
    def inner1(func):
        def inner2(*args, **kwargs):
            experiment_name = os.path.basename(inspect.stack()[1].filename).split(".")[0]
            if subfix:
                experiment_name += "_" + subfix
            run_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            if len(args) == 1 and \
                    hasattr(args[0], "notrain") and \
                    args[0].notrain:
                return func(*args, **kwargs, use_mlflow=True)
            else:
                if mlflow.active_run() is not None:
                    print("There is an active run.")
                else:
                    print("No active run.")

                if mlflow.get_experiment_by_name(experiment_name) is None:
                    mlflow.create_experiment(experiment_name)
                    
                mlflow.set_experiment(experiment_name)

                print(f"experiment_name: {experiment_name} \trun_name:{run_name}")
                with mlflow.start_run(run_name=run_name):
                    # log param
                    for key in kwargs:
                        if "hyperparams" in key and isinstance(kwargs[key], dict):
                            [mlflow.log_param(k, v) for k, v in kwargs[key].items()]

                    if len(args):
                        if is_dataclass(args[0]):
                            [mlflow.log_param(field.name, getattr(args[0], field.name)) for field in fields(args[0])]
                        else:
                            args_dict = vars(args[0])
                            [mlflow.log_param(k, args_dict[k]) for k in args_dict]
                        
                    return func(*args, **kwargs, use_mlflow=True)
        return inner2
    return inner1
