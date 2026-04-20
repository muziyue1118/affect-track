import math
from functools import wraps
from math import floor
from traceback import format_exc
from tqdm import tqdm

import numpy as np
import random
from typing import *

import datetime
import time
from threading import Thread, Lock
import multiprocessing as mp
from skorch.utils import to_numpy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from contextlib import contextmanager
import os
import pickle
import struct
from typing import Dict, Any, List, Optional
import portalocker  # 用于文件锁定

def to_covariance(x):
    lt = x.shape[-1]
    x = x - x.mean(dim=-1, keepdim=True)
    x = x @ x.transpose(-1, -2)
    x = x / lt
    x = (x + x.transpose(-1, -2)) / 2
    return x


def covariance_decompose(cov):
    # Step 1: Compute standard deviations for each channel
    std_dev = (torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))+1e-6)

    # Step 2: Compute correlation matrix
    corr = cov / std_dev.unsqueeze(-1) / std_dev.unsqueeze(-2)
    return std_dev, corr


def batch_fisher_ratio(x1, x2):
    m1 = x1.mean(axis=0)
    v1 = x1.var(axis=0)
    m2 = x2.mean(axis=0)
    v2 = x2.var(axis=0)
    fr = (m1-m2)**2/(v1+v2+1e-6)
    return fr


def manual_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # noqa
    torch.backends.cudnn.benchmark = False  # noqa


class TrainingCheckpoint:
    def __init__(self, name:str):
        self.name = name
        self.file_dir = name

    def _mkdir(self):
        os.makedirs(self.file_dir,exist_ok=True)

    def exists_others(self):
        if os.path.exists(f"{self.file_dir}/others.pt"):
            try:
                torch.load(f"{self.file_dir}/others.pt",map_location="cpu")
            except:
                print("Others File Corrupted")
                return False
            return True
        return False

    def exists(self):
        if os.path.exists(f"{self.file_dir}/model.pt"):
            try:
                torch.load(f"{self.file_dir}/model.pt",map_location="cpu")
                if os.path.exists(f"{self.file_dir}/optim.pt"):
                    torch.load(f"{self.file_dir}/optim.pt", map_location="cpu")
                if os.path.exists(f"{self.file_dir}/others.pt"):
                    torch.load(f"{self.file_dir}/others.pt", map_location="cpu")
                if os.path.exists(f"{self.file_dir}/schedule.pt"):
                    torch.load(f"{self.file_dir}/schedule.pt", map_location="cpu")
            except:
                print("File Corrupted!")
                return False
            return True
        return False

    def save_model(self, model:nn.Module):
        self._mkdir()
        file_path = f"{self.file_dir}/model.pt"
        torch.save(model.state_dict(),file_path)

    def load_model(self, map_location=None):
        if map_location is None:
            map_location =  f"cuda:{torch.cuda.current_device()}"
        self._mkdir()
        file_path = f"{self.file_dir}/model.pt"
        return torch.load(file_path, map_location=map_location)

    def save_optim(self, optim:torch.optim.Optimizer):
        self._mkdir()
        file_path = f"{self.file_dir}/optim.pt"
        torch.save(optim.state_dict(), file_path)

    def load_optim(self, map_location=None):
        # if map_location is None:
        #     map_location = f"cuda:{torch.cuda.current_device()}"
        self._mkdir()
        file_path = f"{self.file_dir}/optim.pt"
        return torch.load(file_path, map_location=map_location)

    def save_others(self, others):
        self._mkdir()
        file_path = f"{self.file_dir}/others.pt"
        torch.save(others, file_path)

    def load_others(self,map_location="cpu"):
        self._mkdir()
        file_path = f"{self.file_dir}/others.pt"
        return torch.load(file_path,map_location=map_location)

    def save_schedule(self, schedule):
        self._mkdir()
        file_path = f"{self.file_dir}/schedule.pt"
        torch.save(schedule.state_dict(), file_path)

    def load_schedule(self, map_location=None):
        if map_location is None:
            map_location =  f"cuda:{torch.cuda.current_device()}"
        self._mkdir()
        file_path = f"{self.file_dir}/schedule.pt"
        return torch.load(file_path, map_location=map_location)

class TransferDataset(Dataset):
    def __init__(self, original_dataset, func):
        self.original_dataset = original_dataset
        self.func = func

    def __getitem__(self, item):
        x, y = self.original_dataset[item]
        return self.func(x), y

    def __len__(self):
        return len(self.original_dataset)

class TransferYDataset(Dataset):
    def __init__(self, original_dataset, func):
        self.original_dataset = original_dataset
        self.func = func

    def __getitem__(self, item):
        x, y = self.original_dataset[item]
        return x, self.func(y)

    def __len__(self):
        return len(self.original_dataset)

class FewChannelDataset(Dataset):
    # Select a few channels manually
    def __init__(self, original_dataset, selected_channels, dim=1):
        self.original_dataset = original_dataset
        self.selected_channels = selected_channels.copy()
        self.dim = dim

    def __getitem__(self, item):
        x, y = self.original_dataset[item]
        if self.dim == 1:
            return x[:,self.selected_channels], y
        else:
            return x[self.selected_channels], y

    def __len__(self):
        return len(self.original_dataset)

class SqueezeDataset(Dataset):
    # Select a few channels manually
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, item):
        X,y = self.original_dataset[item]
        if X.ndim==3:
            return X.reshape(X.shape[1:]), y
        else:
            return X, y

    def __len__(self):
        return len(self.original_dataset)

class UnsqueezeDataset(TransferDataset):
    # Select a few channels manually
    def __init__(self, original_dataset):
        super().__init__(original_dataset,self.transfer)

    @staticmethod
    def transfer(x):
        if x.ndim==2:
            return x[None]
        return x


class RereferenceDataset(Dataset):
    # Select a few channels manually
    def __init__(self, original_dataset, ref_param):
        self.original_dataset = original_dataset
        self.ref_param = ref_param

    def __getitem__(self, item):
        mode = self.ref_param["mode"]
        if mode=="average":
            X,y = self.original_dataset[item]
            meanX = X.mean(axis=0,keepdim=True)
            return X-meanX, y

    def __len__(self):
        return len(self.original_dataset)

class StandardlizeDataset(TransferDataset):
    # 1 std 0 mean at dim -1
    def __init__(self, original_dataset):
        def func(X):
            X = X - X.mean(axis=-1, keepdims=True)
            stds = X.std(axis=-1, keepdims=True)
            stds = abs(stds - 1e-4)+1e-4
            X = X / stds

            return X
        super().__init__(original_dataset, func)


class FewClassesDataset(Dataset):
    # Select a few channels manually
    def __init__(self, original_dataset, selected_classes):
        self.original_dataset = original_dataset
        self.selected_classes = selected_classes
        self.good_samples = []
        self.good_y = []
        for ind,(x,y) in enumerate(original_dataset):
            if y in selected_classes:
                self.good_samples.append(ind)
                self.good_y.append(selected_classes.index(y))

    def __getitem__(self, item):
        if 0<=item<len(self):
            return self.original_dataset[self.good_samples[item]][0],self.good_y[item]
        else:
            raise IndexError()

    def __len__(self):
        return len(self.good_samples)

class CombineClassesDataset(Dataset):
    # Select a few channels manually
    def __init__(self, original_dataset, cls0_list, cls1_list):
        self.original_dataset = original_dataset
        self.cls0_list = cls0_list.copy()
        self.cls1_list = cls1_list.copy()
        self.good_samples = []
        self.good_y = []
        for ind,(x,y) in enumerate(original_dataset):
            if y in cls0_list:
                self.good_samples.append(ind)
                self.good_y.append(0)
            elif y in cls1_list:
                self.good_samples.append(ind)
                self.good_y.append(1)

    def __getitem__(self, item):
        if 0<=item<len(self):
            return self.original_dataset[self.good_samples[item]][0],self.good_y[item]
        else:
            raise IndexError()

    def __len__(self):
        return len(self.good_samples)


class OVRDataset(CombineClassesDataset):
    def __init__(self, original_dataset, target_class, total_class):
        all_classes = list(range(total_class))
        O_classes = [target_class]
        R_classes = [i for i in all_classes if i!=target_class]
        super().__init__(original_dataset, O_classes, R_classes)

class CVSplitDataset(Dataset):
    def __init__(self, original_dataset, as_train, total_cv, select_cv=0, shuffle_seed=0, stratified=True):
        self.original_dataset = original_dataset
        self.total_cv = total_cv
        self.as_train = as_train  # Train: Use all CV splits except select_cv; Test: Select_CV Only
        self.select_cv = select_cv
        self.shuffle_seed = shuffle_seed
        self.stratified = stratified
        total_length = len(original_dataset)
        if not stratified:
            old_state = random.getstate()
            random.seed(shuffle_seed)
            shuffle_ind = list(range(total_length))
            random.shuffle(shuffle_ind)
            random.setstate(old_state)
            start_ind = floor(total_length/total_cv*select_cv)
            end_ind = floor(total_length/total_cv*(select_cv+1))
            test_ind = shuffle_ind[start_ind:end_ind]
            if self.as_train:
                self.selected_ind = [i for i in shuffle_ind if i not in test_ind]
            else:
                self.selected_ind = test_ind
        else:
            kfold = StratifiedKFold(total_cv, shuffle=True, random_state=shuffle_seed)
            shuffle_ind = list(range(total_length))
            ys = [d[1] if isinstance(d[1],(float,int)) else d[1].item() for d in original_dataset]
            all_list = list(kfold.split(shuffle_ind, ys))
            if self.as_train:
                self.selected_ind = all_list[select_cv][0]
            else:
                self.selected_ind = all_list[select_cv][1]
            random.shuffle(self.selected_ind)
        self.selected_length = len(self.selected_ind)

    def __len__(self):
        return self.selected_length

    def __getitem__(self, item):
        return self.original_dataset[self.selected_ind[item]]



class SelectedDataset(Dataset):
    def __init__(self, original_dataset, select_ind):
        self.original_dataset = original_dataset
        self.selected_ind = select_ind
        self.selected_length = len(self.selected_ind)

    def __len__(self):
        return self.selected_length

    def __getitem__(self, item):
        return self.original_dataset[self.selected_ind[item]]


class VRAMDataset(Dataset):
    # Put all the data into VRAM in advanced
    def __init__(self,original_dataset,to_cuda="cuda",verbose=False):
        X = []
        Y = []
        if verbose:
            import tqdm
            original_dataset = [original_dataset[i] for i in range(len(original_dataset))]
            original_dataset = tqdm.tqdm(original_dataset,desc="Loading VRAM Dataset")
        for x,y in original_dataset:
            if not isinstance(x,torch.Tensor):
                X.append(torch.Tensor(x))
            else:
                X.append(x)
            Y.append(int(y))
        X = torch.stack(X).to(to_cuda).float()
        Y = torch.LongTensor(Y).to(to_cuda)  # noqa
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item],self.Y[item]

    def __len__(self):
        return len(self.Y)


class MaxbatchDataloader:
    def __init__(self, max_batch, dl:DataLoader):
        self.max_batch = max_batch
        self.dl = dl
        self.length = max_batch
        if len(dl)<max_batch:
            self.length = len(dl)

    def __len__(self):
        return self.length

    def iter(self):
        dl_iter = iter(self.dl)
        for _ in range(self.length):
            yield next(dl_iter)

    def __iter__(self):
        return self.iter()



class RepeatBatchLoader(MaxbatchDataloader):
    def __init__(self, repeat_num, dl:DataLoader):
        super().__init__(repeat_num, InfinityLoader(dl))

class RandomChangeDataLoader:
    def __init__(self, prob, n_classes, dl:DataLoader):
        self.prob = prob
        self.dl = dl
        self.n_classes = n_classes

    def __len__(self):
        return len(self.dl)

    def iter(self):
        dl_iter = iter(self.dl)
        for _ in range(len(self.dl)):
            X, y = next(dl_iter)
            random_X = torch.normal(0,0.01,X.shape,device=X.device, dtype=X.dtype)
            random_y = torch.randint(0,self.n_classes,y.shape,device=y.device, dtype=y.dtype)
            probs = torch.rand(len(X))
            change_ind = probs<self.prob
            X[change_ind]=random_X[change_ind]
            y[change_ind]=random_y[change_ind]
            yield X, y


    def __iter__(self):
        return self.iter()

class MainSubDataloader:
    def __init__(self, main_DL, sub_DL, prob, probreduce=0.):
        self.main_DL = main_DL
        self.sub_DL = InfinityLoader(sub_DL)
        self.prob = prob
        self.probreduce = probreduce

    def __len__(self):
        return len(self.main_DL)

    def iter(self):
        main_iter = iter(self.main_DL)
        sub_iter = iter(self.sub_DL)
        current_prob = self.prob

        for _ in range(len(self.main_DL)):
            X1,y1 = next(main_iter)
            X2,y2 = next(sub_iter)
            assert len(X1)==len(X2), "Batchsize of Main_DL and Sub_DL should be equal!"
            ind = torch.rand(len(X1), device=X1.device).lt(current_prob)
            current_prob-=self.probreduce
            X = X1.clone()
            Y = y1.clone()
            X[ind]=X2[ind]
            Y[ind]=y2[ind]
            yield X, Y

    def __iter__(self):
        return self.iter()


class ChannelMask:
    def __init__(self, masktype, params=None, dataset=None):
        self.masktype = masktype
        self.params = {} if params is None else params
        if dataset is not None:
            if masktype == "gaussian":
                means = []
                vars = []
                for X,y in dataset:
                    means.append(X.mean(axis=1))
                    vars.append(X.var(axis=1))
                avgmean = sum(means)/len(means)
                avgstd = (sum(vars)/len(vars))**0.5

                if not isinstance(avgmean, torch.Tensor):
                    avgmean = torch.tensor(avgmean)
                    avgstd = torch.tensor(avgstd)
                avgmean = avgmean.float().cuda()
                avgstd = avgstd.float().cuda()
                self.params = {
                    "mean":avgmean,
                    "std": avgstd,
                }

    def mask(self, x, mask):
        # mask: n_batch * n_channel, 0 or 1
        if mask.ndim==1:
            mask=mask[None].repeat(x.shape[0],1)
        assert x.ndim==3
        n_batch = x.shape[0]
        n_channel = x.shape[1]
        n_length = x.shape[2]
        assert mask.shape[0]==n_batch
        assert mask.shape[1]==n_channel
        with torch.no_grad():
            if self.masktype == 'minigaussian':
                x0 = torch.normal(0, 0.01, x.shape, device=x.device, dtype=x.dtype)
            elif self.masktype == "gaussian":
                x0 = torch.normal(0, 1, x.shape, device=x.device, dtype=x.dtype)
                mean = self.params["mean"].reshape(1,-1,1).repeat(n_batch,1,n_length)
                std = self.params["std"].reshape(1,-1,1).repeat(n_batch,1,n_length)
                x0 = x0*std+mean
            elif self.masktype == "permutation":
                x0 = torch.clone(x)
                rand_ind = torch.randperm(n_batch)
                x0 = x0[rand_ind]  # ！：如果一次mask了两个通道，这两个通道的数据将属于同一个样本。
            else:
                raise ValueError(f"Masktype {mask} is not supported!")
            x = x.reshape(n_batch*n_channel,-1)
            x0 = x0.reshape(n_batch*n_channel,-1)
            m = mask.reshape(n_batch*n_channel).bool()
            x[m] = x0[m]
            x = x.reshape(n_batch,n_channel,-1)
            return x

class InfinityLoader:
    def __init__(self, DL):
        self.DL = DL
        self.cur = None
    def __len__(self):
        return 99999999999
    def iter(self):
        if self.cur is None:
            self.cur = iter(self.DL)
        while True:
            try:
                yield next(self.cur)
            except StopIteration:
                self.cur = iter(self.DL)
                yield next(self.cur)
    def __iter__(self):
        return self.iter()

class MaskDataLoader:
    def __init__(self, masktype, dl:DataLoader, dataset=None, params=None):
        self.dl = dl
        if params is None:
            self.params = {
                "mode":"fix",
                "N":1,
            }
        else:
            self.params = params
        self.channel_mask = ChannelMask(masktype,dataset=dataset)

    def __len__(self):
        return len(self.dl)

    def iter(self):
        dl_iter = iter(self.dl)
        params = self.params
        for _ in range(len(self.dl)):
            X, y = next(dl_iter)
            if params["mode"]=="fix":
                N = params["N"]
                mask_prob = torch.rand(X.shape[0], X.shape[1])
                mask_sort = mask_prob.argsort(dim=-1)
                mask_select = mask_sort<N
                mask_map = torch.zeros_like(mask_prob, dtype=torch.long)
                mask_map[mask_select]=1
                X = self.channel_mask.mask(X, mask_map)
            elif params["mode"]=="rand":
                prob = params["prob"]
                mask_prob = torch.rand(X.shape[0], X.shape[1])
                mask_map = torch.zeros_like(mask_prob, dtype=torch.long)
                mask_map[mask_prob<prob]=1
                X = self.channel_mask.mask(X, mask_map)
            elif params["mode"]=="full":
                prob = params["prob"]
                change_prob = random.random()
                if change_prob<prob:
                    mask_map = torch.ones(X.shape[0],X.shape[1],dtype=torch.long)
                else:
                    mask_map = torch.zeros(X.shape[0], X.shape[1], dtype=torch.long)
                X = self.channel_mask.mask(X, mask_map)
            elif params["mode"]=="manual":
                mask_map = params["mask"]
                X = self.channel_mask.mask(X, mask_map)
            elif params["mode"]=="map":
                n_batch, n_channel, n_length = X.shape
                X0 = torch.clone(X)
                ch_ind = list(range(n_channel))
                map_ind = params["map"]
                X[:,ch_ind]=X0[:,map_ind]
            else:
                raise ValueError(f"Unexpected mode: {params['mode']}")

            yield X, y


    def __iter__(self):
        return self.iter()

import traceback

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            super().run()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

class GPUFuckerProcess:
    def __init__(self,func,use_gpu_id=0):
        self.use_gpu_id = use_gpu_id
        self.func = func

    def __call__(self):
        print("GPU FUCKER PROCESS START: USE GPU - ",self.use_gpu_id)
        torch.cuda.set_device(self.use_gpu_id)
        self.func()


class GPUFucker:
    def __init__(self,
                 max_process: Optional[int] = None,
                 refresh_interval: float = 10.,
                 gpu_list:Optional[List[int]]=None):
        if max_process is None:
            self.max_process = 4
        else:
            self.max_process = max_process
        self.refresh_interval = refresh_interval
        self.all_tasks = []
        self.tasks_wait = []
        self.run_cnt = 0
        self.error_cnt = 0
        self.all_cnt = 0
        self.finish_cnt = 0
        self.end_flag = 0
        self.all_finished_is_end = False
        self.thread: Optional[Thread] = None
        if gpu_list is None:
            gpu_list = [0]
        self.gpu_list = gpu_list
        self.gpu_cnt_dict = {
            i:0 for i in gpu_list
        }
        self.threadlock = Lock()

    def write_log(self, *logs):
        tm = time.time()
        tm_str = datetime.datetime.fromtimestamp(tm).strftime("%Y-%m-%d %H:%M:%S.%f")
        logfile = f"fucker_log.txt"
        with open(logfile, "a", encoding="utf-8") as f:
            log_str = ' '.join([str(l) for l in logs])
            f.write(f"[{tm_str}]  {log_str}\n")
        print(log_str)

    def add_task(self, task_func, name: str):
        d = {
            "task_name": name,
            "task_func": task_func,
            "add_time": time.time(),
            "statue": "free",
            "error": None,
        }
        self.tasks_wait.append(d)
        self.all_tasks.append(d)
        self.all_cnt += 1

    def run_task(self, task_dict):
        def func():
            self.threadlock.acquire()
            task_func = task_dict['task_func']
            task_dict['start_time'] = time.time()
            task_dict['statue'] = "run"
            use_gpu = min(self.gpu_cnt_dict,key=lambda x:self.gpu_cnt_dict[x])
            self.gpu_cnt_dict[use_gpu] += 1
            self.write_log("*** RUN *** :", task_dict['task_name'], "Using GPU", use_gpu, "GPU DICT:", str(self.gpu_cnt_dict))
            GFP = GPUFuckerProcess(task_func,use_gpu_id=use_gpu)
            process = Process(target=GFP, name="PROCESS_" + task_dict["task_name"], daemon=False)
            process.start()
            self.threadlock.release()
            process.join()
            self.threadlock.acquire()
            ex = process.exception
            if ex:
                err, trace = ex
                task_dict['statue'] = "error"
                task_dict['end_time'] = time.time()
                task_dict['duration'] = task_dict['end_time'] - task_dict['start_time']
                self.write_log("!!! ERROR !!!  :", task_dict['task_name'], "Duration:", f"{task_dict['duration']:.2f}s")
                self.write_log(f"Error info: {err}")
                self.write_log(f"Trace Back:\n{trace}")
                self.run_cnt -= 1
                self.error_cnt += 1
            else:
                task_dict['statue'] = "finish"
                task_dict['end_time'] = time.time()
                task_dict['duration'] = task_dict['end_time'] - task_dict['start_time']
                self.run_cnt -= 1
                self.finish_cnt += 1
                self.write_log("*** FINISH *** :", task_dict['task_name'], "Duration:", f"{task_dict['duration']:.2f}s", "Progress", self.finish_cnt, "/", self.all_cnt)
            self.gpu_cnt_dict[use_gpu] -= 1
            self.threadlock.release()


        try:
            thread = Thread(target=func, name="THREAD_" + task_dict["task_name"], daemon=True)
            thread.start()

        except Exception as e:
            self.write_log("Thread Error:", e, '\n', format_exc())
            raise e

    def stop(self):
        self.end_flag = 1

    def join(self):
        self.all_finished_is_end = True
        self.thread.join()

    def start(self):
        self.end_flag = 0
        # torch.multiprocessing.set_start_method('spawn')

        def func():
            start_tm = 0
            can_add = False
            while self.end_flag == 0:
                # refresh?
                cur_tm = time.time()
                if self.run_cnt == 0 or cur_tm - start_tm > self.refresh_interval:
                    start_tm = time.time()
                    if self.run_cnt < self.max_process:
                        can_add = True
                if can_add and self.error_cnt == 0:
                    if len(self.tasks_wait) > 0:
                        current_task = self.tasks_wait[0]
                        self.tasks_wait = self.tasks_wait[1:]
                        self.run_cnt += 1
                        self.run_task(current_task)
                        can_add = False
                if self.all_finished_is_end:
                    if self.finish_cnt == self.all_cnt:
                        return 0
                if self.error_cnt > 0:
                    if self.run_cnt == 0:
                        return 1

        self.thread = Thread(target=func, name="GPU_FUCKER_THREAD", daemon=True)
        self.thread.start()

class TableVerbose:
    def __init__(self):
        self.info = {}
        self.order = []
        self.logger = None

    def bind_logger(self, logger=None):
        self.logger = logger

    def add(self, abbr, title, fmt:str="^##"):
        title_length = len(title)
        fmt = fmt.replace("##",str(title_length))
        self.info[abbr] = (title, fmt)
        self.order.append(abbr)

    def print(self,text):
        if self.logger is None:
            print(text)
        elif self.logger is False:
            pass
        else:
            self.logger.info(text)

    def show_title(self):
        titles = ["|"]
        for abbr in self.order:
            titles.append(self.info[abbr][0])
            titles.append("|")
        self.print("".join(titles))

    def show(self, output_info):
        templates = ["|"]
        final_dict = {}
        for abbr in self.order:
            title, fmt = self.info[abbr]
            if abbr in output_info:
                final_dict[abbr] = output_info[abbr]
                templates.append("{")
                templates.append(abbr)
                templates.append(":")
                templates.append(fmt)
                templates.append("}")
                templates.append("|")
            else:
                cur = " "*len(title)
                templates.append(cur)
                templates.append("|")
        template = "".join(templates)
        output = template.format(**final_dict)
        self.print(output)


class BatchStepController:
    def __init__(self,
                 parent_obj,
                 attr_name,
                 start,
                 end,
                 max_epoch,
                 warmstart=0,
                 warmstart_value=None,
                 use_log=True,
                 enable=True,
                 dict_mode=False,):
        self.parent_obj = parent_obj
        self.dict_mode = dict_mode
        self.attr_name = attr_name
        self.start = start
        self.end = end
        self.max_epoch = max_epoch
        self.use_log = use_log
        self.cur_epoch = 0
        self.cur_batch = 0
        self.max_batch = 0
        self.warmstart = warmstart
        self.enable = enable
        if warmstart_value is None:
            self.warmstart_value = start
        else:
            self.warmstart_value = warmstart_value

    def get_value(self):
        if self.dict_mode:
            return self.parent_obj[self.attr_name]
        else:
            return getattr(self.parent_obj, self.attr_name)

    def set_value(self, value):
        if self.use_log:
            if self.dict_mode:
                self.parent_obj[self.attr_name] = 10**value
            else:
                setattr(self.parent_obj,self.attr_name,10**value)
        else:
            if self.dict_mode:
                self.parent_obj[self.attr_name] = value
            else:
                setattr(self.parent_obj,self.attr_name,value)

    def epoch_start(self, cur_epoch, max_batch):
        if not self.enable:
            return
        self.cur_epoch = cur_epoch
        self.cur_batch = 0
        self.max_batch = max_batch
        if cur_epoch < self.warmstart:
            self.set_value(self.warmstart_value)
        else:
            cur_epoch = self.cur_epoch - self.warmstart
            max_epoch = self.max_epoch - self.warmstart
            if cur_epoch>max_epoch:
                cur_epoch = max_epoch
            v = (self.end-self.start)/max_epoch*cur_epoch+self.start
            self.set_value(v)

    def batch_step(self):
        if not self.enable:
            return
        if self.cur_epoch < self.warmstart:
            self.set_value(self.warmstart_value)
        else:
            if self.cur_epoch > self.max_epoch:
                self.set_value(self.end)
                return
            self.cur_batch+=1
            cur_epoch = self.cur_epoch-self.warmstart
            cur_batch = self.cur_batch
            start = self.start
            end = self.end
            max_batch = self.max_batch
            max_epoch = self.max_epoch-self.warmstart
            v_start = (end-start)/max_epoch*cur_epoch+start
            v_end = (end-start)/max_epoch*(cur_epoch+1)+start
            v = (v_end-v_start)/max_batch*cur_batch+v_start
            self.set_value(v)

def calc_weight_decay_loss(network, weight_decay=1.):
    wd_loss = 0
    for name, param in network.named_parameters():
        if 'bn' not in name and 'batchnorm' not in name:
            wd_loss += torch.sum(torch.pow(param, 2))
    return wd_loss*weight_decay

def lst2str_format(lst,fmt,sep=","):
    outputs = []
    for a in lst:
        outputs.append(f"{a:{fmt}}")
    return sep.join(outputs)


def init_handle(addr, name, verbose):
    import logging
    os.makedirs(addr, exist_ok=True)
    file_handle = logging.FileHandler(f"{addr}/{name}.txt",mode="a")
    time_formatter = logging.Formatter('%(asctime)s - %(message)s')
    none_formatter = logging.Formatter('%(message)s')
    file_handle.setFormatter(time_formatter)
    stream_handle = logging.StreamHandler()
    stream_handle.setFormatter(none_formatter)
    LOGGER = logging.getLogger(name)
    for hdlr in LOGGER.handlers[:]:  # remove all old handlers
        LOGGER.removeHandler(hdlr)
    LOGGER.addHandler(file_handle)
    if verbose:
        LOGGER.addHandler(stream_handle)
    LOGGER.setLevel(logging.DEBUG)
    return LOGGER

def calc_fr(F1,F2,axis=-1, oneway=False):
    m1 = F1.mean(axis=axis,keepdims=True)
    m2 = F2.mean(axis=axis,keepdims=True)
    v1 = F1.var(axis=axis,keepdims=True)
    v2 = F2.var(axis=axis,keepdims=True)
    if oneway:
        return (torch.sign(m1-m2)*(m1-m2)**2/(v1+v2)).mean(axis=axis)
    else:
        return ((m1-m2)**2/(v1+v2)).mean(axis=axis)


def GI(ori_M, K, tol=1e-6, maxiter=50):
    eigs = []
    eigvs = []
    for k in range(K):
        M = ori_M
        M = M / M.amax(dim=[-1, -2], keepdims=True)
        last_x = M.sum(dim=-1, keepdims=True)
        for n_iter in range(maxiter):
            M = M @ M
            M = M / M.amax(dim=[-1, -2], keepdims=True)
            x = M.sum(dim=-1, keepdims=True)
            if (x - last_x).abs().max() < tol:
                break
            last_x = x
            # res.append((x.T@M@x).item())
        x = last_x
        xT = x.transpose(-1, -2)
        xTx = xT @ x
        l1 = (xT @ ori_M @ x) / xTx
        # l1 = (last_x.T @ ori_M @ last_x) / (last_x.T @ last_x)
        # l1 = (last_x.T @ A @ last_x) / (last_x.T @ B @ last_x)
        # ori_M = ori_M - l1*(last_x@last_x.T)/(last_x.T@last_x)
        ori_M = ori_M - l1 * (x @ xT) / (xTx)
        eigs.append(l1.squeeze(-1))
        eigvs.append(x)
    # eigs: ... x K
    # eigvs: ... x  n_ch X K
    eigs, eigvs = torch.cat(eigs, dim=-1), torch.cat(eigvs, dim=-1)
    # print(eigvs.shape)
    return eigs, eigvs.transpose(-1, -2)


def solve_GRQ(A, B, method="pi", niter=10, k=1):
    # Solve generalized rayleigh quotient
    if method == "eig":
        assert A.ndim == 3
        n_f, n_ch, _ = A.shape
        n_size = n_f * n_ch
        M = torch.linalg.solve(B, A)
        eig_values, u_mat = torch.linalg.eig(M)
        sort_indices = torch.argsort(eig_values.abs(), descending=True, dim=1)
        indices_fix = torch.arange(n_f, device=A.device) * n_ch
        fixed_sort_indices = (sort_indices + indices_fix[:, None]).reshape(-1)
        u_mat = torch.transpose(u_mat, 1, 2).real
        u_mat = u_mat.reshape(n_size, n_ch)[fixed_sort_indices].reshape(n_f, n_ch, n_ch)
        return u_mat[:, :k]
    elif method == "pi":
        M = torch.linalg.solve(B, A)
        return GI(M, k, 1e-6, niter)[1]

def soft_weight_max(x, tau, dim=-1, keepdim=False):
    # Tau = 1:  mean
    # Tau = -1: max
    mean_x = x.mean(dim=dim, keepdim=True)
    std_x = (x+torch.randn_like(x)*1e-6).std(dim=dim, keepdim=True)
    x = (x - mean_x) / std_x
    wx = torch.softmax(x / (10**tau), dim=dim)
    wmx = torch.sum(wx * x, dim=dim, keepdim=True)
    # wmx = wmx*(xmax-xmin)+xmin
    wmx = wmx * std_x + mean_x
    if keepdim is False:
        wmx = wmx.squeeze(dim)
    return wmx

def soft_average_topk(x, p, k, dim=-1, keepdim=False):
    # p:  0: mean(x)  1: mean_topk(x)
    mean_x = x.mean(dim=dim, keepdim=keepdim)
    klow = math.floor(k)
    khigh = klow+1
    topkhigh = x.topk(khigh, dim=dim)[0]
    topklow = topkhigh.movedim(dim, 0)
    topklow = topklow[:-1]
    topklow = topklow.movedim(0,dim)
    soft_k = k-klow
    mean_topk = topklow.mean(dim=dim, keepdim=keepdim) * (1-soft_k) + topkhigh.mean(dim=dim, keepdim=keepdim) * soft_k
    return mean_x * (1-p) + mean_topk * p



    # return mean_x * (1-p) + mean_topk * p


def stopped_linspace(_s, _e, _l1, _l2, _l0=None):
    if _l0 is not None:
        t1 = torch.linspace(_s, _e, _l1-_l0)
        t0 = torch.ones(_l0)*t1[0]
        t1 = torch.cat([t0,t1])
    else:
        t1 = torch.linspace(_s, _e, _l1)
    t2 = torch.ones(_l2-_l1) * _e
    return torch.cat([t1,t2])

def stopped_logspace(_s, _e, _l1, _l2):
    t1 = torch.logspace(_s, _e, _l1)
    t2 = torch.ones(_l2-_l1) * t1[-1]
    return torch.cat([t1,t2])


class InnerCVSplitDataLoader:
    def __init__(self, TD, inner_cv_group, conclude_class, sample_per, shuffle_seed=114, to="cuda"):
        class_D = {
            cls: FewClassesDataset(TD, [cls])
            for cls in conclude_class
        }
        class_par_D = {
            cls: [
                DataLoader(
                    VRAMDataset(
                        CVSplitDataset(cD, False, inner_cv_group, par_ind, shuffle_seed),
                        to_cuda=to,
                    ),
                    sample_per,
                    shuffle=True,
                )
                for par_ind in range(inner_cv_group)
            ]
            for cls, cD in class_D.items()
        }
        self.class_par_D = class_par_D
        self.sample_per = sample_per

    def sample(self, cls):
        class_par_D = self.class_par_D
        par = class_par_D[cls]
        X = [next(iter(DL))[0] for DL in par]
        X = torch.stack(X, dim=1).reshape(-1, *X[0].shape[1:])
        return X


import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSlider, QWidget, QInputDialog
from PyQt5.QtCore import Qt, QCoreApplication


class SliderPlotWidget(QWidget):
    def __init__(self, func, vmin, vmax, figsize=None, page=False, mode=False, cmd=False):
        super().__init__()
        self.func = func
        self.vmin = vmin
        self.vmax = vmax
        self.usepage = page
        self.page = 0
        self.usemode = mode
        self.mode = ""
        self.usecmd = cmd
        self.cmd = ""

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.canvas = FigureCanvas(self.fig)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.valueChanged.connect(self.update_plot)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.slider)

        self.thread = threading.Thread(target=self.plot_in_thread)
        self.thread.start()

    def plot_in_thread(self):
        if self.usepage:
            self.func(self.vmin, 0)
        else:
            self.func(self.vmin)

    def update_plot(self, value):
        plt.clf()  # Clear the entire figure including subplots
        kwargs = {}
        if self.usepage:
            kwargs["page"] = self.page
        if self.usemode:
            kwargs["mode"] = self.mode
        if self.usecmd:
            kwargs["cmd"] = self.cmd

        self.func(value, **kwargs)
        self.canvas.draw()
        QCoreApplication.processEvents()

    def keyPressEvent(self, event):

        if self.usepage and event.key() in range(Qt.Key_0, Qt.Key_9 + 1):
            self.page = event.key() - Qt.Key_0
            # self.slider.setValue(self.vmin)
            self.update_plot(self.slider.value())
        elif self.usemode and event.key() in range(Qt.Key_A, Qt.Key_Z+1):
            self.mode = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[event.key() - Qt.Key_A]
            self.update_plot(self.slider.value())
        elif self.usecmd and event.key() == Qt.Key_Return:
            self.cmd, ok = QInputDialog.getText(self, 'Input Command', 'Enter command:')
            if ok:
                self.update_plot(self.slider.value())
        else:
            super().keyPressEvent(event)

def slider_plot(func, vmin, vmax, figsize=None, page=False, mode=False, cmd=False):
    app = QApplication([])
    window = QMainWindow()

    slider_plot = SliderPlotWidget(func, vmin, vmax, figsize, page, mode, cmd)

    window.setCentralWidget(slider_plot)
    window.setGeometry(100, 100, 800, 600)
    window.setWindowTitle('Slider Bar Plot')
    window.show()

    try:
        app.exec()
    except KeyboardInterrupt:
        app.quit()

def calc_CE_loss(f, y, already_softmax=False):
    n_classes = f.shape[-1]
    if already_softmax:
        f = torch.log(f)
    else:
        f = torch.log_softmax(f, dim=-1)
    mask = F.one_hot(y, n_classes)
    select_f = f*mask[:,*([None]*(f.ndim-mask.ndim)),:]
    # 3.8:
    # mask_shape = mask.shape
    # new_shape = mask_shape + (1,) * (f.ndim - mask.ndim)
    # mask_expanded = mask.reshape(new_shape)
    # select_f = f * mask_expanded

    select_f = select_f.sum(dim=-1)
    return -select_f


def iselect(source, index, dim=0):
    # 1) If the index is a list, convert it to a tensor
    if isinstance(index, list):
        index = torch.tensor(index, dtype=torch.long, device=source.device)

    # Ensure that the index is on the same device as the source tensor
    index = index.to(source.device)

    # 2) Perform the index_select operation
    return source.index_select(dim, index)

class TensorLoader:
    def __init__(self, X, Y, batch_size, shuffle=False, origin_indices=None, dropend=False, attr=None):
        self.X = X
        if Y is None:
            Y = torch.zeros(len(X), device=X.device)
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0
        self.dropend = dropend
        if origin_indices is None:
            origin_indices = torch.arange(len(X))

        self.length = len(origin_indices)
        self.origin_indices = origin_indices
        self.indices = origin_indices
        self.attr = attr
        
    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def clone(self):
        shuffle = self.shuffle
        self.shuffle = False
        TL = TensorLoader(self.get_X(), self.get_Y(), self.batch_size, shuffle)
        self.shuffle = shuffle
        return TL

    def get_random_remove(self, num_remove, remove_seed, return_removed=False):
        XX = self.get_X()
        YY = self.get_Y()
        generator = torch.Generator().manual_seed(remove_seed)

        classes = torch.unique(YY)
        remaining_indices = []
        removed_indices = []

        for class_label in classes:
            # Get indices of samples with this class
            class_indices = torch.where(YY == class_label)[0]
            # Randomly select indices to remove
            perm = torch.randperm(len(class_indices), generator=generator)
            remove_indices = class_indices[perm[:num_remove]]

            # Keep track of removed indices
            removed_indices.append(remove_indices)
            # Keep the remaining indices
            remaining_indices.append(class_indices[perm[num_remove:]])

        # Concatenate and sort to maintain original order (minus removed samples)
        remaining_indices = torch.cat(remaining_indices).sort().values
        removed_indices = torch.cat(removed_indices).sort().values

        # Filter the original tensors
        XX_remaining = XX[remaining_indices]
        YY_remaining = YY[remaining_indices]

        XX_removed = XX[removed_indices]
        YY_removed = YY[removed_indices]

        if return_removed:
            return (
                TensorLoader(XX_remaining, YY_remaining, self.batch_size, self.shuffle),
                TensorLoader(XX_removed, YY_removed, self.batch_size, self.shuffle)
            )
        else:
            return TensorLoader(XX_remaining, YY_remaining, self.batch_size, self.shuffle)

    def get_X(self):
        # return self.X.index_select(0, self.origin_indices)
        return iselect(self.X, self.origin_indices)
        # return self.X[self.origin_indices]

    def get_Y(self):
        return iselect(self.Y, self.origin_indices)
        # return self.Y[self.origin_indices]

    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        if isinstance(self.attr, torch.Tensor):
            self.attr.to(device)

    def cuda(self, device=None):
        self.X = self.X.cuda(device)
        self.Y = self.Y.cuda(device)
        if isinstance(self.attr, torch.Tensor):
            self.attr.cuda(device)
        return self

    def cpu(self):
        self.X = self.X.cpu()
        self.Y = self.Y.cpu()
        if isinstance(self.attr, torch.Tensor):
            self.attr.cpu()
        return self

    def augment(self, augfunc, augsize, equal=True, C=None, as_XY=False):
        if C is None:
            C = self.Y.amax().item() + 1
        assert augsize%C==0
        augsize_per = augsize//C
        xs = {y:None for y in range(C)}
        shuffle = self.shuffle
        self.shuffle = True
        while sum([0 if x is None else len(x) for x in xs.values()])<augsize:
            x, y = next(iter(self))
            for i in range(C):
                ax = augfunc(x)
                if xs[i] is None:
                    xs[i] = ax[y==i]
                else:
                    xs[i] = torch.cat([xs[i], ax[y==i]], dim=0)
                if equal:
                    xs[i] = xs[i][:augsize_per]

        self.shuffle = shuffle
        Y = torch.tensor(sum([[y]*len(xs[y]) for y in range(C)],[]), device=self.Y.device)
        X = torch.cat([xs[y] for y in range(C)], dim=0)
        if as_XY:
            return X, Y
        else:
            return TensorLoader(
                torch.cat([self.get_X(), X], dim=0),
                torch.cat([self.get_Y(), Y], dim=0),
                self.batch_size,
                self.shuffle,
                dropend=self.dropend,
            )


    def split(self, left_ratio=0.8, seed=0, left_batchsize=None, left_shuffle=None, right_batchsize=None, right_shuffle=None):
        cur_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        left_length = int(self.length*left_ratio)
        randind = self.origin_indices[torch.randperm(self.length)]
        left_ind = randind[:left_length]
        right_ind = randind[left_length:]
        left_TL = TensorLoader(self.X, self.Y,
                               left_batchsize or self.batch_size,
                               left_shuffle or self.shuffle,
                               left_ind)
        right_TL = TensorLoader(self.X, self.Y,
                               right_batchsize or self.batch_size,
                               right_shuffle or self.shuffle,
                               right_ind)
        torch.random.set_rng_state(cur_state)
        return left_TL, right_TL

    def CVsplit(self, total_cv, seed=None):
        kfold = StratifiedKFold(total_cv, shuffle=True, random_state=seed)
        yy = to_numpy(self.Y)
        TDLs = []
        VDLs = []
        for tind, vind in kfold.split(self.origin_indices, yy):
            TDLs.append(TensorLoader(self.X, self.Y, self.batch_size, True, torch.tensor(tind).long()))
            VDLs.append(TensorLoader(self.X, self.Y, self.batch_size, False, torch.tensor(vind).long()))
        return TDLs, VDLs

    def __iter__(self):
        if self.shuffle:
            self.indices = self.origin_indices[torch.randperm(self.length)]
        else:
            self.indices = self.origin_indices
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.length:
            raise StopIteration

        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        if self.dropend and len(batch_indices)<self.batch_size:
            raise StopIteration
        self.current_index += self.batch_size
        if self.attr == "index":
            return (iselect(self.X, batch_indices), batch_indices), iselect(self.Y, batch_indices)
        elif self.attr is not None:
            return (iselect(self.X, batch_indices), iselect(self.attr, batch_indices)), iselect(self.Y, batch_indices)
        else:
            return iselect(self.X, batch_indices), iselect(self.Y, batch_indices)
        # return self.X[batch_indices], self.Y[batch_indices]

    def get_XY(self):
        last_batch = self.batch_size
        self.batch_size = len(self.X)
        x, y = next(iter(self))
        self.batch_size = last_batch
        return x, y


def safelog(x):
    return torch.log(torch.clamp(x, 1e-6, 1e6))

class SafeLog(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return safelog(x)

@contextmanager
def conditional_no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def batch_process(func, X, batch_size=16, no_grad=True):
    with conditional_no_grad(no_grad=no_grad):
        outs = []
        DL = TensorLoader(X, X, batch_size, shuffle=False)
        for x, _ in DL:
            outs.append(func(x))
        return torch.cat(outs)

def batch_process_ex(func, X, batch_size=16):
    boxes = None
    DL = TensorLoader(X, X, batch_size, shuffle=False)
    for x, _ in DL:
        out = func(x)
        if boxes is None:
            boxes = [[] for _ in range(len(out))]
        for box, obj in zip(boxes, out):
            box.append(obj)
    boxes = [torch.cat(box) for box in boxes]
    return boxes

def PCA_reduce(f, select_component, detail=False):
    ori_f = f
    f = f.movedim(0, -1)
    mu = f.mean(dim=-1, keepdim=True)
    f = f-mu
    f_cov = (f@f.transpose(-1,-2))/(f.shape[-1]-1)
    f_eigv, f_eigm = torch.linalg.eigh(f_cov)
    select_eigv = f_eigv[..., -select_component:]
    select_eigm = f_eigm[..., :, -select_component:]
    f = f.movedim(-1, 0)
    recon_f = (f[..., None, :]@select_eigm)[..., 0, :]
    if not detail:
        return recon_f

    # Variance explained
    total_variance = f_eigv.sum(dim=-1)
    explained_variance = select_eigv.sum(dim=-1)
    explained_variance_ratio = explained_variance / total_variance

    # Validate reconstruction
    reconstructed_f = (recon_f[..., None, :] @ select_eigm.transpose(-1, -2))[..., 0, :] + mu[..., 0]

    # Reconstruction error
    reconstruction_error = torch.mean((ori_f - reconstructed_f) ** 2)
    return recon_f, explained_variance_ratio, reconstruction_error


# def calc_CSP_filter(C1, C2, k, already_mean=False, **kwargs):
#     if already_mean is False:
#         mean_C1 = C1.mean(dim=0)
#         mean_C2 = C2.mean(dim=0)
#     else:
#         mean_C1, mean_C2 = C1, C2
#     w1 = solve_GRQ(mean_C1, mean_C2, k=k, **kwargs)
#     w2 = solve_GRQ(mean_C2, mean_C1, k=k, **kwargs)
#     ws = torch.cat([w1,w2], dim=-2)
#     return ws
#
# def apply_csp_filter(C, ws, norm=True, log=True):
#     # C:  n_b x ... x ch x ch
#     # ws: ... x 2k x n_ch
#     F = ws[...,:,None,:]@C[:,...,None,:,:]@ws[...,:,:,None]
#     F = F[...,0,0]
#     if log:
#         F = torch.log(F)
#     if norm:
#          F = F/F.sum(dim=-1, keepdim=True)
#     return F

def tensor_select(x, ind, dim):
    x = torch.movedim(x, dim, -1)
    e = torch.eye(x.shape[-1], device=x.device)
    onehot = e[ind]
    x = (x*onehot).sum(dim=-1)
    return x


def one_hot(tensor, dim):
    # Get the indices of the maximum values along the specified dimension
    indices = torch.argmax(tensor, dim=dim)

    # Create a one-hot encoded tensor
    one_hot_tensor = torch.eye(tensor.size(dim), device=tensor.device)[indices]

    # Move the one-hot dimension to the specified position
    one_hot_tensor = one_hot_tensor.movedim(-1, dim)

    return one_hot_tensor



def estimate_pickle_size(obj):
    return len(pickle.dumps(obj))

def calc_CE_bayes_risk_loss(evidences, labels):
    alphas = evidences + 1.0
    strengths = torch.sum(alphas, dim=-1, keepdim=True)

    loss = torch.sum(labels * (torch.digamma(strengths) - torch.digamma(alphas)), dim=-1)

    return loss  # n_batch x ...

def calc_KL_div_loss(evidences, labels):
    num_classes = evidences.size(-1)
    alphas = evidences + 1.0
    alphas_tilde = labels + (1.0 - labels) * alphas
    strength_tilde = torch.sum(alphas_tilde, dim=-1, keepdim=True)

    # lgamma is the log of the gamma function
    first_term = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(evidences.new_tensor(num_classes, dtype=torch.float32))
            - torch.sum(torch.lgamma(alphas_tilde), dim=-1, keepdim=True)
    )
    second_term = torch.sum(
        (alphas_tilde - 1.0) * (torch.digamma(alphas_tilde) - torch.digamma(strength_tilde)), dim=-1, keepdim=True
    )
    loss = first_term + second_term

    return loss.mean(dim=-1)  # n_batch x ...


class CEBayesRiskLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Bayes risk is the maximum cost of making incorrect estimates, taking a cost function assigning a penalty of
        making an incorrect estimate and summing it over all possible outcomes. Here the cost function is the Cross Entropy.
        """
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        return calc_CE_bayes_risk_loss(evidences, labels)


class KLDivergenceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        """Acts as a regularization term to shrink towards zero the evidence of samples that cannot be correctly classified"""
        super().__init__(*args, **kwargs)

    def forward(self, evidences, labels):
        return calc_KL_div_loss(evidences, labels)

def calc_EDL_loss(x, labels, coef):
    onehot_y = F.one_hot(labels, num_classes=x.shape[-1])
    onehot_y = onehot_y[:, *[None, ] * (x.ndim - labels.ndim - 1)]
    x = F.softplus(x)  # x to evidences
    loss1 = calc_CE_bayes_risk_loss(x, onehot_y)
    loss2 = calc_KL_div_loss(x, onehot_y)
    return loss1 + loss2 * coef


def shuffle_combine(DL1, DL2, random_state=114, cuda=False):
    # Do copy.
    TX, TY = DL1.X, DL1.Y
    EX, EY = DL2.X, DL2.Y
    AX = torch.concat([TX, EX]).numpy()
    AY = torch.concat([TY, EY]).numpy()
    stkfold = StratifiedKFold(2, shuffle=True, random_state=random_state)
    tind, eind = list(stkfold.split(AX, AY))[0]
    TX = torch.FloatTensor(AX[tind])
    EX = torch.FloatTensor(AX[eind])
    TY = torch.LongTensor(AY[tind])
    EY = torch.LongTensor(AY[eind])
    TDL = TensorLoader(TX, TY, 32, True)
    EDL = TensorLoader(EX, EY, 32, False)
    if cuda:
        TDL = TDL.cuda()
        EDL = EDL.cuda()
    return TDL, EDL

def shuffle_combine_ex(DL1, DL2, n_splits=2, swap_train_test=False, random_state=114, cuda=False, batch_size=32, return_ind=False):
    # Do copy.
    TX, TY = DL1.X, DL1.Y
    EX, EY = DL2.X, DL2.Y
    AX = torch.concat([TX, EX]).numpy()
    AY = torch.concat([TY, EY]).numpy()
    stkfold = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    tind, eind = list(stkfold.split(AX, AY))[0]
    if swap_train_test:
        tind, eind = eind, tind
    TX = torch.FloatTensor(AX[tind])
    EX = torch.FloatTensor(AX[eind])
    TY = torch.LongTensor(AY[tind])
    EY = torch.LongTensor(AY[eind])
    TDL = TensorLoader(TX, TY, batch_size, True)
    EDL = TensorLoader(EX, EY, batch_size, False)
    if cuda:
        TDL = TDL.cuda()
        EDL = EDL.cuda()
    if return_ind:
        return TDL, EDL, tind, eind
    else:
        return TDL, EDL


def singular_value_bounding(w, eps):
    # w: ... x M x N
    # Target: Singular of w is cropped to 1/(1+e) to 1+e
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    # U @ diag_embed(S) @ Vh = w
    S = S.clamp(1 / (1 + eps), 1 + eps)
    w = U @ torch.diag_embed(S) @ Vh
    return w

def integrated_gradients(model, input_tensor, baseline, steps=50):
    """
    Compute Integrated Gradients (IG) for a given model and input.

    Args:
        model: The model to explain.
        input_tensor: Input tensor for which IG is computed.
        baseline: Baseline tensor (usually zeros or neutral input).
        steps: Number of steps for Riemann approximation of the integral.

    Returns:
        Integrated Gradients for the input tensor.
    """
    # Scale inputs from baseline to the input
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)  # Combine inputs for batch processing

    # Enable gradient computation
    scaled_inputs.requires_grad = True

    # Compute model outputs
    outputs = model(scaled_inputs)

    # Compute gradients of outputs w.r.t. inputs
    grads = torch.autograd.grad(
        outputs, scaled_inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]

    # Approximate integral using Riemann summation
    avg_grads = grads[:-1].mean(dim=0)  # Average over steps
    integrated_grads = (input_tensor - baseline) * avg_grads

    return integrated_grads


def plot_angle_histogram(vectors, ax=None, n_bins=36, color="C0"):
    """
    绘制二维向量的角度直方图，并以二维向量形式显示。

    参数：
        vectors (ndarray): 二维数组，形状为 (N, 2)，表示 N 个二维向量。
        n_bins (int): 将角度范围 [0, 360) 划分为的区间数，默认为 12。

    输出：
        极坐标形式的直方图，其中线段长度和宽度表示每个区间的频率。
    """
    if ax is None:
        ax = plt.gca()
    vectors = to_numpy(vectors)
    # 计算每个向量的角度（0-360）
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360

    # 划分角度范围，并计算频率
    bin_edges = np.linspace(0, 360, n_bins + 1)
    freq, _ = np.histogram(angles, bins=bin_edges)

    ax.set_aspect('equal')

    # 绘制线段
    for i in range(n_bins):
        angle = (bin_edges[i] + bin_edges[i + 1]) / 2  # 每个区间的中间角度
        length = freq[i] / max(freq) * 0.9  # 根据频率调整长度（归一化到 [0, 1]）
        width = freq[i] / max(freq) * 5  # 根据频率调整宽度
        x_end = length * np.cos(np.radians(angle))
        y_end = length * np.sin(np.radians(angle))

        ax.plot([0, x_end], [0, y_end], linewidth=width, color=color)

def DL_cov_trace_normalize(TDL, EDL):
    eye = torch.eye(TDL.X.shape[-1], device=TDL.X.device)
    mean_trace = (TDL.X * eye).sum(dim=[-1,-2], keepdim=True).mean(dim=0)
    TDL.X = TDL.X / mean_trace
    EDL.X = EDL.X / mean_trace

def DL_cov_to_corr(DL):
    DL.X = covariance_decompose(DL.X)[1]
    return DL


def generate_normal_matrix(N, C):
    # V = torch.rand(1, 1, 1, N) * torch.eye(N)
    P = torch.randn(N, C, C)
    Pu, _, Pv = torch.linalg.svd(P)
    P = Pu @ Pv
    # X = P @ V @ P.transpose(-1, -2)
    return P

def generate_SPD_matrix(N, C):
    V = torch.rand(N, 1, C) * torch.eye(C)
    P = torch.randn(N, C, C)
    Pu, _, Pv = torch.linalg.svd(P)
    P = Pu @ Pv
    X = P @ V @ P.transpose(-1, -2)
    return X

def square_to_triu(X):
    indx, indy = torch.triu_indices(X.shape[-1], X.shape[-1], offset=0)
    return X[..., indx, indy]

def square_to_vector(X):
    shape = list(X.shape)
    shape[-2] = shape[-2]*shape[-1]
    shape = shape[:-1]
    X = X.reshape(*shape)
    return X

class Reshaper(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    def forward(self, x):
        return x.reshape(self.target_shape)


def calc_CSP_filter(m0, m1, k=1, niter=50, eps=0.):
    if eps>0:
        eps = torch.eye(m0.shape[-1], device=m0.device)*eps
    w0 = solve_GRQ(m0, m1+eps, niter=niter, k=k)
    w1 = solve_GRQ(m1, m0+eps, niter=niter, k=k)
    return torch.cat([w0, w1], dim=-2)

def apply_CSP_filter(ws, X, do_log=True, select_k=None):
    if select_k is not None:
        n_s = ws.shape[-2]
        n_half = n_s//2
        ws = torch.cat([ws[..., :select_k, :], ws[..., n_half:n_half+select_k, :]], dim=-2)
    f = ws[..., None, :]@X[...,None,:,:]@ws[..., None]
    f = f[..., 0, 0]
    if do_log:
        f = safelog(f)
    return f

def calc_robust_mean(X0, X1, robustiter=1, keep_percent=0.8, k=1, niter=10, return_mask=False, keep_percent_decrease=0.):
    # X: n_b x ... x n_c x n_c
    # mask: n_b x ... x 1 x 1
    mask0 = 1.
    mask1 = 1.

    m0 = X0.mean(dim=0)
    m1 = X1.mean(dim=0)
    # m: ... x n_c x n_c
    for n_iter in range(robustiter):
        ws = calc_CSP_filter(m0, m1, k=k, niter=niter)
        F0 = apply_CSP_filter(ws, X0, do_log=True)
        F1 = apply_CSP_filter(ws, X1, do_log=True)
        # F: n_b x ... x n_s
        mF0 = F0.mean(dim=0)
        mF1 = F1.mean(dim=0)
        d0 = (F0 - mF0).norm(dim=-1, p=2)
        d1 = (F1 - mF1).norm(dim=-1, p=2)
        r0 = d0.argsort(dim=0).argsort(dim=0).float()/len(d0)
        r1 = d1.argsort(dim=0).argsort(dim=0).float()/len(d1)
        # r: n_b x ...,  0 .. N-1
        mask0 = (r0<keep_percent).float()[..., None, None]
        mask1 = (r1<keep_percent).float()[..., None, None]
        m0 = (X0 * mask0).sum(dim=0)/(len(d0)*keep_percent)
        m1 = (X1 * mask1).sum(dim=0)/(len(d1)*keep_percent)
        keep_percent -= keep_percent_decrease
    if return_mask:
        return mask0, mask1, m0, m1
    else:
        return m0, m1


def calc_lda_filter_pre(X, Y):
    """
    Pre-calculation for multi-view LDA.

    Computes necessary statistics by removing the batch dimension, and reshapes them to the original view dimensions.
    Returns:
        stats (dict): Dictionary containing:
            - m0 (torch.Tensor): Mean for class 0, reshaped to (*views, n_f)
            - m1 (torch.Tensor): Mean for class 1, reshaped to (*views, n_f)
            - S_w (torch.Tensor): Within-class scatter matrix per view, reshaped to (*views, n_f, n_f)
    """
    n_b = X.shape[0]
    view_shape = X.shape[1:-1]  # extra dimensions (views)
    n_f = X.shape[-1]

    # Compute total number of view elements
    N = 1
    for d in view_shape:
        N *= d

    # Reshape X to (n_b, N, n_f) so that each view is processed independently.
    X_flat = X.reshape(n_b, N, n_f)

    # Boolean masks for the two classes (assumed to be 0 and 1)
    mask0 = (Y == 0)
    mask1 = (Y == 1)

    # Separate samples by class
    X0 = X_flat[mask0]  # shape: (n0, N, n_f)
    X1 = X_flat[mask1]  # shape: (n1, N, n_f)

    # Compute class means for each view (shape: (N, n_f))
    m0 = X0.mean(dim=0)
    m1 = X1.mean(dim=0)

    # Compute within-class scatter matrices for each view without using unsqueeze(0)
    # d0: (n0, N, n_f) and we compute outer products for each view element:
    d0 = X0 - m0  # m0 broadcasts to (n0, N, n_f)
    S0 = (d0[..., None] * d0[..., None].transpose(-1, -2)).sum(dim=0)  # shape: (N, n_f, n_f)

    d1 = X1 - m1
    S1 = (d1[..., None] * d1[..., None].transpose(-1, -2)).sum(dim=0)  # shape: (N, n_f, n_f)

    # Total within-class scatter for each view
    S_w = S0 + S1  # shape: (N, n_f, n_f)

    # Reshape m0, m1 and S_w back to the original view dimensions.
    m0 = m0.reshape(*view_shape, n_f)
    m1 = m1.reshape(*view_shape, n_f)
    S_w = S_w.reshape(*view_shape, n_f, n_f)

    stats = {
        "m0": m0,
        "m1": m1,
        "S_w": S_w,
    }
    return stats


def calc_lda_filter_after(stats):
    """
    Post-calculation for multi-view LDA.

    Computes the LDA filter w and bias b from pre-computed statistics that are already reshaped
    to the original view dimensions.

    Args:
        stats (dict): Dictionary containing:
            - m0 (torch.Tensor): Mean for class 0, shape (*views, n_f)
            - m1 (torch.Tensor): Mean for class 1, shape (*views, n_f)
            - S_w (torch.Tensor): Within-class scatter matrices, shape (*views, n_f, n_f)

    Returns:
        tuple: (w, b)
            - w (torch.Tensor): LDA filter of shape (*views, n_f)
            - b (torch.Tensor): Bias term of shape (*views,)
    """
    m0 = stats["m0"]  # shape: (*views, n_f)
    m1 = stats["m1"]  # shape: (*views, n_f)
    S_w = stats["S_w"]  # shape: (*views, n_f, n_f)

    # Compute the difference between class means for each view.
    diff = (m1 - m0)[..., None]  # shape: (*views, n_f, 1)

    # Add a small regularization term for numerical stability.
    n_f = m0.shape[-1]
    reg = 1e-4 * torch.eye(n_f, device=S_w.device)
    # Solve for w using the LDA formulation: S_w * w = diff, then remove the extra dimension.
    w = torch.linalg.solve(S_w + reg, diff)[..., 0]  # shape: (*views, n_f)

    # Compute the projected means using the precomputed m0 and m1.
    proj0 = (m0 * w).sum(dim=-1)  # shape: (*views,)
    proj1 = (m1 * w).sum(dim=-1)  # shape: (*views,)
    pdiff = (proj1 - proj0).clamp(min=1e-5)

    # Compute the scaling factor per view so that the difference becomes 2.
    a = 2.0 / pdiff  # shape: (*views,)

    # Incorporate the scaling into w.
    w = a[..., None] * w  # shape: (*views, n_f)

    # Compute the bias term so that the normalized projected mean for class 0 becomes -1.
    b = -1.0 - a * proj0  # shape: (*views,)

    return w, b


def calc_lda_filter(X, Y):
    """
    Full calculation of multi-view LDA filter and bias.

    First computes the necessary statistics and then calculates w and b.
    """
    stats = calc_lda_filter_pre(X, Y)
    return calc_lda_filter_after(stats)

def calc_lda_filter_parallal(Xs, Ys):
    m0 = []
    m1 = []
    S_w = []
    for X, Y in zip(Xs, Ys):
        stats = calc_lda_filter_pre(X, Y)
        m0.append(stats['m0'])
        m1.append(stats['m1'])
        S_w.append(stats['S_w'])
    m0 = torch.stack(m0, dim=0)
    m1 = torch.stack(m1, dim=0)
    S_w = torch.stack(S_w, dim=0)
    return calc_lda_filter_after({
        "m0": m0,
        "m1": m1,
        "S_w": S_w,
    })

# def calc_lda_filter(X, Y):
#     """
#     Calculate 2-class LDA (Fisher's linear discriminant) filters with output normalization.
#
#     The function computes a filter w and a bias b such that:
#         - The LDA output: p = (X * w).sum(dim=-1)
#         - The normalized output: p_norm = p + b
#     where the normalized output satisfies:
#         mean(p_norm) for samples with Y==0 is -1, and
#         mean(p_norm) for samples with Y==1 is  1.
#
#     Parameters:
#         X (torch.Tensor): Input tensor of shape (n_b, *views, n_f), where
#                           n_b is the number of samples, *views are any additional dimensions,
#                           and n_f is the input features.
#         Y (torch.Tensor): 1D tensor of shape (n_b,) containing class labels (0 and 1).
#
#     Returns:
#         tuple: (w, b)
#             - w (torch.Tensor): LDA filter of shape (*views, n_f)
#             - b (torch.Tensor): Bias term of shape (*views,)
#     """
#     # Get basic shapes and flatten the view dimensions.
#     n_b = X.shape[0]
#     view_shape = X.shape[1:-1]  # extra dimensions (views)
#     n_f = X.shape[-1]
#
#     # Compute the total number of view elements, N.
#     N = 1
#     for d in view_shape:
#         N *= d
#     # Reshape X to (n_b, N, n_f) so that each "view" is processed independently.
#     X_flat = X.reshape(n_b, N, n_f)
#
#     # Boolean masks for the two classes (assumed to be 0 and 1)
#     mask0 = (Y == 0)
#     mask1 = (Y == 1)
#
#     # Separate samples by class
#     X0 = X_flat[mask0]  # shape: (n0, N, n_f)
#     X1 = X_flat[mask1]  # shape: (n1, N, n_f)
#
#     # Compute means for each view and feature for the two classes
#     m0 = X0.mean(dim=0)  # shape: (N, n_f)
#     m1 = X1.mean(dim=0)  # shape: (N, n_f)
#
#     # Compute within-class scatter matrices for each view
#     d0 = X0 - m0.unsqueeze(0)  # shape: (n0, N, n_f)
#     S0 = torch.einsum('bni,bnj->nij', d0, d0)  # shape: (N, n_f, n_f)
#
#     d1 = X1 - m1.unsqueeze(0)  # shape: (n1, N, n_f)
#     S1 = torch.einsum('bni,bnj->nij', d1, d1)  # shape: (N, n_f, n_f)
#
#     # Total within-class scatter for each view
#     S_w = S0 + S1  # shape: (N, n_f, n_f)
#
#     # Compute the difference between class means (for each view)
#     diff = (m1 - m0).unsqueeze(-1)  # shape: (N, n_f, 1)
#
#     # Solve for w using the LDA formulation: S_w * w = diff
#     # This gives a solution that is defined up to a multiplicative constant.
#     w = torch.linalg.solve(S_w + 1e-4 * torch.eye(S_w.shape[-1], device=S_w.device), diff).squeeze(-1)  # shape: (N, n_f)
#
#     # Compute the projected means for each class using the precomputed means m0 and m1.
#     proj0 = (m0 * w).sum(dim=-1)  # shape: (N,)
#     proj1 = (m1 * w).sum(dim=-1)  # shape: (N,)
#
#     # LDA filter w is scale invariant; we choose to scale it so that the difference
#     # between the class projected means becomes exactly 2. Then a bias b can shift the means to -1 and 1.
#     # Calculate the scaling factor a per view:
#     a = 2.0 / (proj1 - proj0)  # shape: (N,)
#
#     # Update the filter: incorporate the scaling into w.
#     w = a.unsqueeze(-1) * w  # shape: (N, n_f)
#
#     # Compute the bias b per view directly using the projected mean for class 0:
#     b = -1.0 - a * proj0  # shape: (N,)
#
#     # Reshape w and b back to the original view dimensions.
#     w = w.reshape(*view_shape, n_f)  # shape: (*views, n_f)
#     b = b.reshape(*view_shape)         # shape: (*views,)
#
#     return w, b


def apply_lda_filter(X, w, b):
    return (X*w).sum(dim=-1)+b

class LFUDict:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = {}
        self.freq = {}
        self.min_pair = (None, 0)

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None

    def _update_min_pair(self):
        key = min(self.data, key=self.freq.get, default=None)
        freq = self.freq[key]
        return key, freq

    def update(self, key, value):
        if key in self.data:
            self.freq[key] += 1
        else:
            if key not in self.freq:
                self.freq[key] = 0
            self.freq[key] += 1
            cur_freq = self.freq[key]
            if len(self.data)<self.max_length:
                self.data[key] = value
            else:
                # One should be removed, or this should not be added
                if self.min_pair[0] is None:
                    self.min_pair = self._update_min_pair()
                if cur_freq <= self.min_pair[1]:
                    # This should not be added
                    pass
                else:
                    del self.data[self.min_pair[0]]
                    self.data[key] = value
                    self.min_pair = self._update_min_pair()


class SequenceOptimizer:
    def __init__(self, optimizers:List[torch.optim.Optimizer]):
        self.opts = optimizers

    def step(self):
        for opt in self.opts:
            opt.step()

    def zero_grad(self,set_to_none=True):
        for opt in self.opts:
            opt.zero_grad(set_to_none)

def flatten(x):
    x = x.reshape(len(x), -1)
    return x

def retry(max_retry=3, delay=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retry):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retry - 1:
                        raise e  # 最后一次失败直接抛出异常
                    if delay is None:
                        time.sleep(random.random()+0.5)
                    else:
                        time.sleep(delay)
            return None
        return wrapper
    return decorator

class ZipStorage:
    """
    支持快速查找和追加的存储类

    参数:
        file_addr: 文件路径
        MAX_INDEX_SIZE: 最大索引大小(KB)，默认1024KB(1MB)
        lock: 是否启用文件锁，默认True
    """

    def __init__(self, file_addr: str, MAX_INDEX_SIZE: int = 1024, lock: bool = True):
        self.file_addr = file_addr
        self.MAX_INDEX_SIZE = MAX_INDEX_SIZE * 1024  # 转换为字节
        self.lock = lock

    def create(self):
        """创建一个空存档"""
        self.save({})

    def init(self):
        dirname = os.path.dirname(self.file_addr)
        if dirname!="":
            os.makedirs(os.path.dirname(self.file_addr), exist_ok=True)
        if not os.path.exists(self.file_addr):
            self.create()


    @retry(3, None)
    def save(self, res_dict: Dict[str, Any]):
        """
        将字典保存到文件

        参数:
            res_dict: 要保存的字典
        """
        # 准备索引和数据
        index = {}
        data_parts = []
        current_offset = 0

        # 构建索引和数据块
        for key, value in res_dict.items():
            serialized_data = pickle.dumps(value)
            data_len = len(serialized_data)
            index[key] = (current_offset, data_len)
            data_parts.append(serialized_data)
            current_offset += data_len

        # 序列化索引
        serialized_index = pickle.dumps(index)
        index_len = len(serialized_index)

        if index_len > self.MAX_INDEX_SIZE:
            raise ValueError(f"Index size {index_len} exceeds maximum {self.MAX_INDEX_SIZE} bytes")

        # 写入文件
        mode = 'wb'
        if self.lock:
            with portalocker.Lock(self.file_addr, mode='wb', timeout=5) as f:
                self._write_data(f, index_len, serialized_index, data_parts)
        else:
            with open(self.file_addr, 'wb') as f:
                self._write_data(f, index_len, serialized_index, data_parts)

    def _write_data(self, f, index_len, serialized_index, data_parts):
        """实际写入数据的辅助方法"""
        f.write(struct.pack('Q', index_len))
        f.write(serialized_index)
        f.write(b'\x00' * (self.MAX_INDEX_SIZE - index_len))  # 填充预留空间

        # 写入所有数据块
        for data in data_parts:
            f.write(data)

    @retry(3, None)
    def append(self, key: str, value: Any):
        """
        向文件中追加一条记录

        参数:
            key: 要添加的键
            value: 要添加的值

        异常:
            ValueError: 如果键已存在或索引将超过最大大小
        """
        try:
            if self.lock:
                with portalocker.Lock(self.file_addr, mode='rb+', timeout=5) as f:
                    self._do_append(f, key, value)
            else:
                with open(self.file_addr, 'rb+') as f:
                    self._do_append(f, key, value)
        except ValueError:
            cp = self.load_all()
            cp[key] = value
            self.save(cp)

    def _do_append(self, f, key: str, value: Any):
        """实际执行追加操作的辅助方法"""
        # 1. 读取现有索引
        index_len_bytes = f.read(8)
        if len(index_len_bytes) != 8:
            raise ValueError("Invalid file format")
        index_len = struct.unpack('Q', index_len_bytes)[0]

        # 读取索引
        serialized_index = f.read(index_len)
        index = pickle.loads(serialized_index)

        # 检查键是否已存在
        if key in index:
            raise ValueError(f"Key '{key}' already exists in file")

        # 2. 获取当前数据结束位置
        f.seek(0, 2)  # 移动到文件末尾
        new_data_offset = f.tell() - (8 + self.MAX_INDEX_SIZE)

        # 3. 序列化新数据
        serialized_data = pickle.dumps(value)
        data_len = len(serialized_data)

        # 4. 更新索引
        index[key] = (new_data_offset, data_len)
        new_serialized_index = pickle.dumps(index)
        new_index_len = len(new_serialized_index)

        if new_index_len > self.MAX_INDEX_SIZE:
            raise ValueError(f"Updated index size {new_index_len} exceeds maximum {self.MAX_INDEX_SIZE} bytes")

        # 5. 写入新数据
        f.write(serialized_data)

        # 6. 更新索引
        f.seek(0)
        f.write(struct.pack('Q', new_index_len))
        f.write(new_serialized_index)

    def load(self, key: str) -> Any:
        """
        从文件中加载指定键的值

        参数:
            key: 要加载的键

        返回:
            对应的值

        异常:
            KeyError: 如果键不存在
        """
        with open(self.file_addr, 'rb') as f:
            # 读取索引长度
            index_len_bytes = f.read(8)
            if len(index_len_bytes) != 8:
                raise ValueError("Invalid file format")
            index_len = struct.unpack('Q', index_len_bytes)[0]

            # 读取索引
            serialized_index = f.read(index_len)
            index = pickle.loads(serialized_index)

            # 检查键是否存在
            if key not in index:
                raise KeyError(f"Key '{key}' not found in file")

            # 定位并读取数据
            offset, length = index[key]
            f.seek(8 + self.MAX_INDEX_SIZE + offset)
            serialized_data = f.read(length)

            return pickle.loads(serialized_data)

    def load_all(self) -> Dict[str, Any]:
        """
        加载文件中的所有数据

        返回:
            包含所有数据的字典
        """
        result = {}
        with open(self.file_addr, 'rb') as f:
            # 读取索引
            index_len = struct.unpack('Q', f.read(8))[0]
            index = pickle.loads(f.read(index_len))

            # 读取每个数据项
            for key, (offset, length) in index.items():
                f.seek(8 + self.MAX_INDEX_SIZE + offset)
                serialized_data = f.read(length)
                result[key] = pickle.loads(serialized_data)

        return result

    def keys(self) -> List[str]:
        """
        获取文件中所有键的列表

        返回:
            键列表
        """
        with open(self.file_addr, 'rb') as f:
            index_len = struct.unpack('Q', f.read(8))[0]
            index = pickle.loads(f.read(index_len))
            return list(index.keys())

    def exist(self, key: str) -> bool:
        """
        检查键是否存在

        参数:
            key: 要检查的键

        返回:
            如果存在返回True，否则False
        """
        if not os.path.exists(self.file_addr):
            return False

        with open(self.file_addr, 'rb') as f:
            index_len_bytes = f.read(8)
            if len(index_len_bytes) != 8:
                return False
            index_len = struct.unpack('Q', index_len_bytes)[0]

            try:
                index = pickle.loads(f.read(index_len))
                return key in index
            except:
                return False

if __name__ == "__main__":
    # 创建存储实例
    storage = ZipStorage("data.pt", lock=True)
    # 创建空存档
    storage.create()

    # 添加数据
    storage.append("A", {"value": 1})
    storage.append("B", {"value": 2})

    # 检查存在性
    print(storage.exist("A"))  # True
    print(storage.exist("C"))  # False

    # 获取单个值
    print(storage.load("A"))  # {'value': 1}

    # 获取所有键
    print(storage.keys())  # ['A', 'B']

    # 获取所有数据
    print(storage.load_all())  # {'A': {'value': 1}, 'B': {'value': 2}}

    # 保存新数据(会覆盖)
    storage.save({"C": 3, "D": 4})
    print(storage.keys())  # ['C', 'D']

def count_netpara(net):
    return sum([torch.prod(torch.tensor(v.shape)).item() for v in net.parameters()])