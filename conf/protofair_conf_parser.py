DEF_EVAL_BATCH_SIZE = 8
DEF_TRAIN_BATCH_SIZE = 512
DEF_EVAL_NUM_WORKERS = 2
DEF_TRAIN_NUM_WORKERS = 6
DEF_INIT_STD = .1
DEF_ETA_MIN = 1e-6
DEF_DELTA_ON = 'user'
DEF_WD = 1e-5


def parse_conf(conf: dict, ) -> dict:
    """
    Placing default parameters if not present in the configuration file
    :param conf:
    :return:
    """
    if 'eval_batch_size' not in conf:
        conf['eval_batch_size'] = DEF_EVAL_BATCH_SIZE
    if 'train_batch_size' not in conf:
        conf['train_batch_size'] = DEF_TRAIN_BATCH_SIZE

    if 'init_std' not in conf:
        conf['init_std'] = DEF_INIT_STD
    if 'eta_min' not in conf:
        conf['eta_min'] = DEF_ETA_MIN
    if 'delta_on' not in conf:
        conf['delta_on'] = DEF_DELTA_ON
    if 'wd' not in conf:
        conf['wd'] = DEF_WD

    if 'running_settings' not in conf:
        conf['running_settings'] = dict()

    if 'eval_n_workers' not in conf['running_settings']:
        conf['running_settings']['eval_n_workers'] = DEF_EVAL_NUM_WORKERS
    if 'train_n_workers' not in conf['running_settings']:
        conf['running_settings']['train_n_workers'] = DEF_TRAIN_NUM_WORKERS

    return conf
