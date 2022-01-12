# © 2021 Nokia
#
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import time

from .print_ import padSpaces

class Logger():
    def __init__(self, metrics_list, epoch_digits, out_folder, append = False, log_name = 'log.txt'):
        self.metrics_list = metrics_list
        self.epoch_digits = epoch_digits
        self.out_folder   = out_folder
        self.log_name     = log_name
        
        if not append:
            with open(self.out_folder + '/' + self.log_name, 'w') as log:
                metrics_list = metrics_list.copy()
                
                log_header_str = f'epoch,{metrics_list.pop(0)},'
                for m_name in metrics_list:
                    log_header_str += f'{m_name},'

                log_header_str += "time\n"
                
                log.write(log_header_str)

        self._clear()
        
        
    def _clear(self):
        self.time_start = time.time()
        self.metrics    = [[] for _ in range(len(self.metrics_list))]
        
    def accumulate(self, partial_metrics_list, idx = None):
        for i, m in enumerate(partial_metrics_list):
            if idx is None:
                self.metrics[i].append(m)
            else:
                self.metrics[idx[i]].append(m)
            
    
    def log(self, epoch, append = True, **kwargs):
        epoch_str = padSpaces(epoch, self.epoch_digits)
        
        log_entry_str = f'{epoch},'
        log_print_str = f'{epoch_str} '
        
        for i, m in enumerate(self.metrics):
            if len(m) > 0:
                m_avg = sum(m) / len(m)
            else:
                m_avg = -1.0
            
            log_entry_str += f'{m_avg},'
            log_print_str += f' {self.metrics_list[i]}: {m_avg:.4f},'
            
        log_entry_str += f'{time.time() - self.time_start:.4f}\n'
        log_print_str += f' time: {time.time() - self.time_start:.1f}'

        if append:
            with open(self.out_folder + '/'+ self.log_name, 'a') as log:
                log.write(log_entry_str)
            
        print(log_print_str)
        
        self._clear()
