from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC
import traceback



if __name__ == '__main__':
    all_cp = []
    for file in os.listdir('/media/ubuntu/0B1314D80B1314D8/LYH/11.5/ceshi'):
        cp = os.path.join('/media/ubuntu/0B1314D80B1314D8/LYH/11.5/ceshi', file)
        all_cp.append(cp)

    all_cp.sort(key=lambda x: int(x.split('_e')[1][:-4]))

    '''
    'scale_step': 1.0375,
    'scale_lr': 0.59,
    'scale_penalty': 0.9745,
    'window_influence': 0.176,
    
    wi:0.25
    sp:0.96
    
    st:1.0375
    sl:0.59
    '''

    for net_path in all_cp:
        for st in [1.0375]:
            for sl in [0.59]:
                cfg = {'window_influence':0.21,
                       'scale_penalty':0.945,
                       'scale_step':1.0575,
                       'scale_lr':0.59}
                tracker = TrackerSiamFC(net_path=net_path,**cfg)

                e = ExperimentVOT(root_dir='/media/ubuntu/0B1314D80B1314D8/LYH/VOT18/sequences18', version=2018, experiments=('supervised'),
                          read_image=True,
                          result_dir='results',
                          result_suffix=net_path.split('/')[7].split('.')[0] +
                                                                '_wi_' + str(0.25) + '_sp_' + str(0.96))
                e.run(tracker)



