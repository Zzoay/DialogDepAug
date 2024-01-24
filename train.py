
from typing import *
import random
import datetime

import os
import yaml
import wandb
import torch
from torch.utils.data import DataLoader, Subset
import transformers
from transformers import AutoTokenizer

from trainer import InterTrainer
from model import DepParser, InterParser
from utils import arc_rel_loss, uas_las, to_cuda, seed_everything

from data_helper import SynDataset, DialogUttrDataset, InterDataset, DialogUttrInterDataset
from evaluation import eval_eduwise, eval_uttrwise, eval_all, eval_all_new


def run():
    from config import DataClassUnpack, CFG
    # load_config
    with open('config/inter.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    CFG = DataClassUnpack.instantiate(CFG, config)
    CFG.debug = False

    wandb.init(project="Dialogue-Dep-Aug", entity="zzoay")
    if 'alpha' in wandb.config:
        del CFG.alpha
        print('Duplicate parameters deleted.')
    if 'aug_files' in wandb.config:
        del CFG.aug_files
        print('Duplicate parameters deleted.')
    wandb.config.update(CFG)
    CFG.aug_files = wandb.config.aug_files

    tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
    special_tokens_dict = {'additional_special_tokens': ['[Q]', '[A]']}
    num_added_tokens = tokenizer.add_tokens(special_tokens_dict['additional_special_tokens'])
    print('token added', num_added_tokens, 'token')
    CFG.tokenizer = tokenizer

    # traindataset = DialogUttrDataset(CFG, CFG.train_file, tokenizer)
    traindataset = DialogUttrInterDataset(CFG, CFG.train_file, tokenizer)
    syndataset = SynDataset(CFG, tokenizer)

    syn_iter = DataLoader(syndataset, batch_size=CFG.syn_batch_size, shuffle=True, drop_last=True)

    one_epoch_steps = int(len(syndataset) / CFG.syn_batch_size)

    # alignment
    if not CFG.debug:
        n = int(one_epoch_steps / len(traindataset) * CFG.dialog_batch_size)
        rst_ids = list(range(len(traindataset)))
        tmp_ids = []
        for i in range(n):
            tmp_ids.extend(rst_ids)

        remain_num = one_epoch_steps - int(n * len(traindataset) / CFG.dialog_batch_size)
        random.shuffle(rst_ids)
        tmp_ids.extend(rst_ids[:remain_num])
        traindataset_extend = Subset(traindataset, tmp_ids)
        train_iter = DataLoader(traindataset_extend, batch_size=CFG.dialog_batch_size, shuffle=True)

    print("Train Size: " + str(len(traindataset_extend)))

    model = InterParser(CFG)
    wandb.watch(model)

    trainer = InterTrainer(model, one_epoch_steps=one_epoch_steps, loss_fn=arc_rel_loss, metrics_fn=uas_las, config=CFG)

    best_res, best_state_dict = trainer.train(model, train_iter, syn_iter)

    torch.save(best_state_dict, 'ckpt/inter_par.pt')

    # diagdataset_test = DialogEvalDataset(CFG, CFG.test_file, tokenizer)
    # test_iter = DataLoader(diagdataset_test, batch_size=1, shuffle=False)
    # (inner_uas, inner_las), (inter_uas, inter_las) = eval_eduwise(model, CFG, test_iter)
    # print(inner_uas, inner_las), (inter_uas, inter_las)
    # wandb.log({"edu-wise": {"test_inner_uas":inner_uas, "test_inner_las":inner_las,
    #                         "test_inter_uas":inter_uas, "test_inter_las":inter_las}})

    # diagdataset_test = DialogUttrDataset(CFG, CFG.test_file, tokenizer, test=True)
    diagdataset_test = DialogUttrInterDataset(CFG, CFG.test_file, tokenizer, test=True)
    test_iter = DataLoader(diagdataset_test, batch_size=1, shuffle=True)

    inter_dataset = InterDataset(CFG)
    inter_iter = DataLoader(inter_dataset, batch_size=1)
    (inner_uas, inner_las), (inter_uas, inter_las) = eval_all_new(model, test_iter, inter_iter)
    wandb.log({"inner-EDU": {"uas": inner_uas, "las": inner_las}, 
               "inter-EDU": {"uas": inter_uas, "las": inter_las}})
    print((inner_uas, inner_las), (inter_uas, inter_las))
    

if __name__ == '__main__':
    transformers.logging.set_verbosity_error() # only report errors.

    import time
    # time.sleep(5500)
    # import GPUtil
    # import random
    # while True:
    #     time.sleep(random.uniform(420, 600))

    #     # 获取当前GPU的负载信息
    #     GPUs = GPUtil.getGPUs()
    #     if len(GPUs) > 0:
    #         gpu = GPUs[0]  # 假设你想监控第一个GPU
    #         gpu_load = gpu.load * 100  # 获取GPU占用率

    #         # 打印当前GPU占用率（可选）
    #         print(f"Current GPU load: {gpu_load}%")

    #         # 如果GPU占用率低于10%，则跳出循环
    #         if gpu_load < 10:
    #             print("GPU load is below 10%, exiting the loop.")
    #             break
    #     else:
    #         print("No GPU found.")
    #         break
    
    seed_everything(42)

    debug = False
    use_wandb = True
    sweep = False

    # wandb
    if debug:
        os.environ['WANDB_MODE'] = 'disabled'  # offline / disabled

    if not use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'  # offline / disabled
    
    time_now = datetime.datetime.now().isoformat()
    print(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')

    if sweep:
        sweep_configuration = {
            "name": "aug-sweep",
            "metric": {"name": "inter-EDU.las", "goal": "maximize"},
            "method": "grid",
            "parameters": {
                "aug_files": {
                    "values": [
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json'],
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data-extend/aug_by_word_1.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json', 'aug-data-extend/aug_utterance_1.json'],
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data-extend/aug_by_word_1.json', 'aug-data-extend/aug_by_word_2.json', 'aug-data-extend/aug_by_word_3.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json', 'aug-data-extend/aug_utterance_1.json', 'aug-data-extend/aug_utterance_2.json', 'aug-data-extend/aug_utterance_3.json'],
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json', 'aug-data/aug_uttr_free.json', 'aug-data-extend/aug_uttr_free_0.json'],
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data-extend/aug_by_word_1.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json', 'aug-data-extend/aug_utterance_1.json', 'aug-data/aug_uttr_free.json', 'aug-data-extend/aug_uttr_free_0.json', 'aug-data-extend/aug_uttr_free_1.json'],
                        ['aug-data/aug_word_uttr.json', 'aug-data-extend/aug_by_word_0.json', 'aug-data-extend/aug_by_word_1.json', 'aug-data-extend/aug_by_word_2.json', 'aug-data-extend/aug_by_word_3.json', 'aug-data/aug_rpl_utterance.json', 'aug-data-extend/aug_utterance_0.json', 'aug-data-extend/aug_utterance_1.json', 'aug-data-extend/aug_utterance_2.json', 'aug-data-extend/aug_utterance_3.json', 'aug-data/aug_uttr_free.json', 'aug-data-extend/aug_uttr_free_0.json', 'aug-data-extend/aug_uttr_free_1.json', 'aug-data-extend/aug_uttr_free_2.json', 'aug-data-extend/aug_uttr_free_3.json'],
                        # ['zero/zero-uttr.json'],
                    ]
                }
            }
        }
        sweep_id = wandb.sweep(sweep_configuration, project="Dialogue-Dep-Aug", entity="zzoay")
        # run the sweep
        wandb.agent(sweep_id, function=(run))
    else:
        run()

    print('=================End=================')
    print(datetime.datetime.now().isoformat())
    print('=====================================')