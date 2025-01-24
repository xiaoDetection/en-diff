from mmcv import runner
from mmcv.runner.hooks import HOOKS,Hook, LrUpdaterHook
from typing import List

@HOOKS.register_module()
class TrainModeControlHook(Hook):
    def __init__(self, train_modes, num_epoch, alt_modes=None):
        self.train_modes = train_modes
        self.swich_epoch = []
        self.alt_modes = alt_modes

        for i, e in enumerate(num_epoch):
            self.swich_epoch.append(e + self.swich_epoch[i - 1] if i > 0 else e)

    def before_run(self, runner):
        self.model = runner.model.module

    def before_train_epoch(self, runner):
        epoch = runner.epoch + 1
        train_stage = 0
        while epoch > self.swich_epoch[train_stage] and train_stage < len(self.train_modes) - 1:
            train_stage += 1

        train_mode = self.train_modes[train_stage]
        if train_mode == 'alt':
            train_mode = self.alt_modes[(epoch - self.swich_epoch[train_stage] - 1) % len(self.alt_modes)]
            self.model.train_mode_control(train_mode)
        else:
            self.model.train_mode_control(train_mode)
        
        runner.logger.info('train mode: %s'%train_mode)
        

@HOOKS.register_module()
class MulStepLrUpdaterHook(LrUpdaterHook):
    def __init__(
            self,
            step,
            lr_mul: List[float],
            min_lr = None,
            warmup_start: int = 0,
            **kwargs,
        ):
        """
            warmup_start: warmup start point, 1 start
        """
        super().__init__(**kwargs)
        self.step = step
        self.min_lr = min_lr
        self.lr_mul = [1] + lr_mul
        self.warmup_start = warmup_start

    def get_lr(self, runner: runner.BaseRunner, base_lr: float):
        progress = runner.epoch if self.by_epoch else runner.iter

        lr_mul = self.lr_mul[-1]
        for i, s in enumerate(self.step):
            if progress < s:
                lr_mul = self.lr_mul[i]
                break

        lr = base_lr * lr_mul
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        return lr
    
    def before_train_epoch(self, runner: 'runner.BaseRunner'):
        super().before_train_epoch(runner)

        # warmup: [start, end)
        if self.by_epoch:
            epoch_len = len(runner.data_loader)
            self.warmup_start_iter = (self.warmup_start - 1) * epoch_len
        else:
            self.warmup_start_iter = self.warmup_start - 1
        self.warmup_end_iter = self.warmup_start_iter + self.warmup_iters

    def before_train_iter(self, runner: 'runner.BaseRunner'):
        cur_iter = runner.iter

        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_end_iter or cur_iter < self.warmup_start_iter:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_iter = cur_iter - self.warmup_start_iter
                warmup_lr = self.get_warmup_lr(warmup_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_end_iter or cur_iter < self.warmup_start_iter:
                return
            elif cur_iter == self.warmup_end_iter:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_iter = cur_iter - self.warmup_start_iter
                warmup_lr = self.get_warmup_lr(warmup_iter)
                self._set_lr(runner, self.regular_lr)
