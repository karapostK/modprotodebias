import glob
import json
import os
import socket
from pathlib import Path

import wandb
from paramiko import SSHClient
from scp import SCPClient

from conf.conf_parser import parse_conf_file


def fetch_best_in_sweep(sweep_id: str, wandb_entity_name: str = None, wandb_project_name: str = None,
                        good_faith: bool = True, project_base_directory: str = '.'):
    """
    Utility function to fetch the configuration of the best model from a sweep. It assumes that the sweep has been run
    with method = 'bayes'. It also adapts the paths to the local directory.

    If good_faith is False, it will connect to the wandb API, fetch the configuration, and possibly import the model
    from one of the servers.
    If good_faith is True, it will look for the model in the local directory. It will assume that for a specific sweep id
    there only one model (i.e. the best one). If there are more than one, it will raise an error.

    :param sweep_id: The id of the sweep
    :param wandb_entity_name: The name of the wandb entity
    :param wandb_project_name: The name of the wandb project
    :param good_faith: Whether to trust that the local directory has the best model
    :param project_base_directory: The base directory of the project. It will be used to store the model.
    """

    if "PycharmProjects" in os.getcwd():
        preamble_path = '~/PycharmProjects'
    else:
        preamble_path = '~'

    if good_faith:

        glob_path = f'{project_base_directory}/saved_models/*/sweeps/{sweep_id}'

        glob_results = glob.glob(glob_path)
        if len(glob_results) == 0:
            raise FileNotFoundError(f'The sweep was not found in the local directory {glob_path}')
        elif len(glob_results) > 1:
            raise ValueError('There should not be two sweeps with the same id')

        sweep_path = glob_results[0]
        best_run_paths = os.listdir(sweep_path)

        if len(best_run_paths) == 0:
            raise FileNotFoundError(f'The sweep does not contain any runs')
        elif len(best_run_paths) > 1:
            raise ValueError('There are more than 1 runs in the project, which one is the best?')

        best_run_path = best_run_paths[0]
        best_run_config = parse_conf_file(os.path.join(sweep_path, best_run_path, 'conf.yml'))

    else:
        api = wandb.Api()
        sweep = api.sweep(f"{wandb_entity_name}/{wandb_project_name}/{sweep_id}")

        best_run = sweep.best_run()

        best_run_host = best_run.metadata['host']
        best_run_config = json.loads(best_run.json_config)

        if '_items' in best_run_config:
            best_run_config = best_run_config['_items']['value']
        else:
            best_run_config = {k: v['value'] for k, v in best_run_config.items()}

        best_run_path = best_run_config['model_path']
        print('Best Run Model Path: ', best_run_path)

        best_run_local_path = os.path.join(project_base_directory, best_run_path)

        current_host = socket.gethostname()

        if not os.path.isdir(best_run_local_path):
            Path(best_run_local_path).mkdir(parents=True, exist_ok=True)

            if current_host != best_run_host:
                print(f'Importing Model from {best_run_host}')
                # Moving the best model to local directory
                # N.B. Assuming same username
                with SSHClient() as ssh:
                    ssh.load_system_host_keys()
                    ssh.connect(best_run_host)

                    with SCPClient(ssh.get_transport()) as scp:
                        # enoughcool4hardcoding
                        dir_path = "hassaku"
                        if best_run_host == 'passionpit.cp.jku.at':
                            dir_path = os.path.join(dir_path, "PycharmProjects")

                        scp.get(remote_path=os.path.join(dir_path, best_run_path),
                                local_path=os.path.dirname(best_run_local_path),
                                recursive=True)
            else:
                raise FileNotFoundError(
                    f"The model should be local but it was not found! Path is: {best_run_local_path}")

    # Adapt Dataset path
    if preamble_path:
        pre, post = best_run_config['dataset_path'].split('hassaku/', 1)
        best_run_config['dataset_path'] = os.path.join(preamble_path, 'protofair', post)
        pre, post = best_run_config['data_path'].split('hassaku/', 1)
        best_run_config['data_path'] = os.path.join(preamble_path, 'protofair', post)

    # Running from non-main folder
    best_run_config['model_save_path'] = os.path.join(project_base_directory, best_run_config['model_save_path'])
    best_run_config['model_path'] = os.path.join(project_base_directory, best_run_config['model_path'])
    return best_run_config
