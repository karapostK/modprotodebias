DEF_EVAL_BATCH_SIZE = 8
DEF_TRAIN_BATCH_SIZE = 512
DEF_EVAL_NUM_WORKERS = 6
DEF_TRAIN_NUM_WORKERS = 6
DEF_INIT_STD = .1
DEF_ETA_MIN = 1e-6
DEF_DELTA_ON = 'user'
DEF_WD = 1e-5
DEF_GRADIENT_SCALING = 1
DEF_USE_CLAMPING = True


def parse_conf(conf: dict, ) -> dict:
    """
    Placing default parameters if not present in the configuration file
    :param conf:
    :return:
    """

    added_parameters_list = []

    if 'eval_batch_size' not in conf:
        conf['eval_batch_size'] = DEF_EVAL_BATCH_SIZE
        added_parameters_list.append(f"eval_batch_size={conf['eval_batch_size']}")
    if 'train_batch_size' not in conf:
        conf['train_batch_size'] = DEF_TRAIN_BATCH_SIZE
        added_parameters_list.append(f"train_batch_size={conf['train_batch_size']}")

    if 'init_std' not in conf:
        conf['init_std'] = DEF_INIT_STD
        added_parameters_list.append(f"init_std={conf['init_std']}")
    if 'eta_min' not in conf:
        conf['eta_min'] = DEF_ETA_MIN
        added_parameters_list.append(f"eta_min={conf['eta_min']}")
    if 'delta_on' not in conf:
        conf['delta_on'] = DEF_DELTA_ON
        added_parameters_list.append(f"delta_on={conf['delta_on']}")
    if 'wd' not in conf:
        conf['wd'] = DEF_WD
        added_parameters_list.append(f"wd={conf['wd']}")
    if 'gradient_scaling' not in conf:
        conf['gradient_scaling'] = DEF_GRADIENT_SCALING
        added_parameters_list.append(f"gradient_scaling={conf['gradient_scaling']}")
    if 'use_clamping' not in conf:
        conf['use_clamping'] = DEF_USE_CLAMPING
        added_parameters_list.append(f"use_clamping={conf['use_clamping']}")

    if 'running_settings' not in conf:
        conf['running_settings'] = dict()

    if 'eval_n_workers' not in conf['running_settings']:
        conf['running_settings']['eval_n_workers'] = DEF_EVAL_NUM_WORKERS
        added_parameters_list.append(f"eval_n_workers={conf['running_settings']['eval_n_workers']}")
    if 'train_n_workers' not in conf['running_settings']:
        conf['running_settings']['train_n_workers'] = DEF_TRAIN_NUM_WORKERS
        added_parameters_list.append(f"train_n_workers={conf['running_settings']['train_n_workers']}")

    print(f"Added parameters: {', '.join(added_parameters_list)}")

    return conf
