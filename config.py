from pprint import pprint
from pprint import pformat
import os
import logging
# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    data_dir = './gtx980-high-dvfs-real-small-workload-features.csv'
    out_pred_dir = '/home/zhtang/water/txt/'
    out = 'predict'
    # for transfomers
    norm_mean = 0.0
    norm_std = 1.0

    # pretrained
    pretrained = None

    # architecture of network
    customize = True
    arch = 'gpu_net_13'

    train_num_workers = 8
    test_num_workers = 8

    # optimizers
    optim = 'SGD'
    use_adam = False

    # param for optimizer
    lr = 0.001
    weight_decay = 0.00005
    lr_decay = 0.33  #

    # record i-th log
    kind = '0'

    # set gpu :
    # gpu = True

    plot_every = 10
    # training
    epoch = 120

    # if eval
    evaluate = False

    # debug
    # debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None
    save_path = '~/water/modelparams'

    train_begin = 0
    train_end = 524

    test_begin = 525
    test_end = 749

    columns_13 = list(range(2, 15))
    columns_39 = list(range(17, 56))
    # columns = (379, 385, 390, 391, 392, 406, 414, 415, 416, 417, 418, 419, 420, 422,
    # 425, 434, 435, 436, 438, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449,
    # 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 464, 465, 466, 468, 512,
    # 513, 514, 515, 517, 518, 519, 520, 557, 558, 559, 560, 561, 562
    # )

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
        if opt.customize:
            logging_name = 'log' + '_self_' + opt.arch + '_' + opt.optim + opt.kind + '.txt' 
        else:
            logging_name = 'log' + '_default_' + opt.arch + '_' + opt.optim + opt.kind + '.txt'
        if not os.path.exists('log'):
            os.mkdir('log')

        if opt.arch == 'gpu_net_13':
            self.columns = self.columns_13
        elif opt.arch == 'gpu_net_39':
            self.columns = self.columns_39
        logging_path = os.path.join('log', logging_name)
    
        logging.basicConfig(level=logging.DEBUG,
                        filename=logging_path,
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
        logging.info('Logging for {}'.format(opt.arch))
        logging.info('======user config========')
        logging.info(pformat(self._state_dict()))
        logging.info('==========end============')
        # logging.info('optim : [{}], batch_size = {}, lr = {}, weight_decay= {}, momentum = {}'.format( \
        #                 args.optim, args.batch_size,
        #                 args.lr, args.weight_decay, args.momentum) )






    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
    

opt = Config()
