
import json
import random
import time
import re

from tqdm import tqdm
import openai

from constant import *

openai.api_base = api_base
openai.api_key = api_key


def aug_word_utterance(start_idx):
    def generate_slots(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_lst =  []
        
        for d in data:
            rel_dct = {}
            for tripple in d['relationship']:
                head, rel, tail = tripple
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
                if head_uttr_idx != tail_uttr_idx:
                    continue
                
                if not rel_dct.get(head_uttr_idx, None):
                    rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
                else:
                    rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
                
            for item in d['dialog']:
                turn = item['turn']
                utterance = item['utterance']
                idx_lst, word_lst = [], []

                if len(utterance.split(' ')) < 5:
                    sample_lst.append([[], utterance])
                    continue
                
                for word_idx, word in enumerate(utterance.split(' ')):
                    head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                    word_lst.append(f'{word}')
                sample_lst.append([idx_lst, ' '.join(word_lst)])
        return sample_lst  

    new_instances = []
    for uttr_id, (ids, instance) in enumerate(tqdm(generate_slots('data/zero.json'))):
        if uttr_id < start_idx:
            continue
        prompt = """
        你具有一定的语言学背景，精通中文文本理解，尤其是依存分析。
        给定一个文本，请进行词语替换（严格地一一对应）。
        请一步一步的进行思考。
        请遵循如下步骤：
        1. 首先，解析给定文本的句法结构，找出句子中的谓词。 
        2. 以谓词为中心，对句子进行改写，但不能改变词语的顺序。
        请注意如下几点：
        - 不能改变原始的句法结构和词序，但是整体语义可以改变。
        - 使用中文，并遵循电商客服和用户对话的文本风格。
        - 改写后的文本中的词语（包括标点）应该与原始文本中的词语严格对应。
        - 尽可能多地替换词语，重点关注谓词。
        - 输出一个答案后结束。
        - 以空格为分词，不要改变原始输入的分词。
        举例来说，输入: 这 件 商品 我 要 退 了 ，明天 取 件 。\n
        输出: 
        谓词: 退、取
        替换后: 这 件 衣服 我 要 买 了 ，后天 取 件 。
        """
        if len(instance) > 150:
            instance = instance[:150]
        
        prompt += f"Origin: {instance}\nOutput:"
        
        if len(instance.split(' ')) > 4:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",  top_p=0.5, n=5, messages=[{"role": "user", "content": prompt}])
            for i in range(5):
                answer = completion.choices[i].message.content
                new_instance = answer[answer.rindex('替换后')+len('替换后')+1:].strip()
                if '\n' in new_instance:
                    new_instance = new_instance.split('\n')[1]
                with open(f'aug/aug_by_word_{i}.txt', 'a', encoding='utf-8') as f:
                    f.write(new_instance + '\n')
        else:
            for i in range(5):
                with open(f'aug/aug_by_word_{i}.txt', 'a', encoding='utf-8') as f:
                    f.write(instance + '\n')

def aug_utterance(start_idx=0):
    def generate_sample(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sample_lst =  []
        
        for d in data:
            rel_dct = {}
            for tripple in d['relationship']:
                head, rel, tail = tripple
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
                if head_uttr_idx != tail_uttr_idx:
                    continue
                
                if not rel_dct.get(head_uttr_idx, None):
                    rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
                else:
                    rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
            
            for item in d['dialog']:
                turn = item['turn']
                utterance = item['utterance']
                tripples = []

                if len(utterance.split(' ')) < 5:
                    sample_lst.append([[], utterance])
                    continue

                word_lst = list(utterance.split(' '))
                
                for word_idx, word in enumerate(word_lst):
                    head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct']) 
                    if rel == 'root':
                        tripples.append([f'{head_word_idx}-root', rel2name[rel], f'{word_idx + 1}-{word}'])
                    elif rel not in relsyn2id.keys():
                        tripples.append([f'{head_word_idx}-{word_lst[head_word_idx-1]}', rel2name[rel], f'{word_idx + 1}-{word}'])

                sample_lst.append([tripples, '[seg]'.join(word_lst)])
        return sample_lst 
    
    for uttr_id, (tripples, instance) in enumerate(tqdm(generate_sample('data/zero.json'))):
        if uttr_id < start_idx:
            continue
        prompt = """
        你具有一定的语言学背景，精通中文文本理解，尤其是依存分析。
        给定一个文本，将他进行改写，改写的文本的语篇结构与原文相似。
        Let's think step by step. 
        首先分析给定文本的语篇结构，然后将其改写成与原文相似的语篇结构。
        请注意： 
        1. 形式为， Origin: [given text], Ouput:  
        2. 使用中文，遵循电商客服和用户对话的文本风格。
        3. 将修辞结构理论（rhetorical structure theory）作为文本语篇结构。
        4. 改写后的文本长度与原文尽可能相等。
        5. 考虑基本话语单元EDU（elementary discourse units)，并且不要改变EDU的顺序。
        6. 改写后的文本中的词语应该用[seg]进行分隔，与原始文本一致（细粒度的中文分词）。
        7. 不要对给定文本进行任何回复或续写，只需改写原文。
        举例来说, Origin: [这个[seg]商品[seg]我[seg]要[seg]退[seg]了[seg]，[seg]由于[seg]质量[seg]实在[seg]太差]
        Output: 这件[seg]衣服[seg]我[seg]想[seg]买[seg]了[seg]，[seg]因为[seg]版型[seg]确实[seg]很好
        """
        if len(instance) > 150:
            instance = instance[:150].strip()
        instance = instance.replace(' ', '[seg]')
        prompt += f"Origin: [{instance}] \n Output:"
        
        if len(instance.split('[seg]')) > 2 and random.random() < 0.3:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",  top_p=0.5, n=4,messages=[{"role": "user", "content": prompt}])
            
            for i in range(4):
                answer = completion.choices[i].message.content
                new_instance = answer.strip()
                instance = new_instance
                if '\n' in instance:
                    instance = instance.split('\n')[1]
                with open(f'aug/aug_utterance_{i}.txt', 'a', encoding='utf-8') as f:
                    f.write(instance.replace('[seg]', ' ') + '\n')
        else:
            for i in range(4):
                with open(f'aug/aug_utterance_{i}.txt', 'a', encoding='utf-8') as f:
                    f.write(instance.replace('[seg]', ' ') + '\n')

def aug_utterance_free(start_idx=0):
    rels = []
    with open('relation.csv', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            rels.append('|'.join(line.strip().split(',')))
    rels = '\n '.join(rels)
    template = """
    你具有一定的语言学背景，精通中文文本理解，尤其是依存分析。
    请遵循如下步骤：
    1. 对于给定的文本及其篇章关系，请解释当前篇章关系和目标篇章关系。 
    2. 对于指定的核心，根据新的篇章关系，列举出有意义的依附示例。 
    3. 对于指定的依附，根据新的篇章关系，列举出有意义的核心示例。 
    4. 根据第二步和第三步中的依存组合示例，对输入文本进行重写，并按照指定格式输出。
    5. 对于多个篇章关系，重复上述步骤，并最终输出为一个完整的改写后文本。
    请严格遵守如下约束：
    1. 使用中文并遵循原始文本的风格. 
    2. 上下文逻辑应当合理. 
    3. 不要对给定文本进行回复或续写
    4. 输出格式：文本`\{1\}'，核心`\{2\}'，依附`\{3\}'，推理步骤`\{4\}
    请仿照如下的例子：
    输入: 文本"如果 这 件 商品 明天 还 没 到 货 ， 我 要 退 了"，核心"我 要 退 了"，依附"如果 这件 商品 明天 还 没 到 货"，当前篇章关系"条件"，目标篇章关系"因果"
    输出: 
    推理步骤:
    (1)篇章关系'条件'是指依附为核心的前提，而篇章关系'因果'是指依附为核心的原因。
    (2)"我 要 退 了"，对于篇章关系因果，依附的例子可以为"商品 不 符合 购买 的 需求”，"这 件 商品 没 到 货"，"商品 无法 正常 使用”。
    (3)"如果 这 件 商品 明天 还 没 到 货"，对于篇章关系'因果'核心的例子可以为"我 明天 无法 使用 该 商品”，"我 准备 购买 其它 商品”，"我 要 退 款"。
    (4)综上依存组合，根据"这件 商品 没 到 货"和"我 要 退 款”，生成了当前新的文本。
    因此，生成的结果：
    核心"我 要 退 了"，依附"因为 这件 商品 没 到 货"，文本"因为 这件 商品 没 到 货 ， 我 要 退 款"
    """
    with open('data/zero.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for uttr_id, (word_lst, split_ids, inter_rels) in enumerate(tqdm(edu_gen(data))):
        if uttr_id < start_idx:
            continue
        
        edus = []
        for i, split_id in enumerate(split_ids):
            if i == 0:
                continue
            edus.append(' '.join(word_lst[split_ids[i - 1]:split_id]))

        inter_rels = [[x[0], rst_dct[x[1]], x[2]] for x in inter_rels]
        if len(inter_rels) == 0:
            inter_rels = [inter_rels]
        
        uttr = ' '.join(edus)
        if len(uttr) > 150:
            uttr = uttr[:150]
        
        if inter_rels[0] != []:
            instance = " "
            for i, inter_rel in enumerate(inter_rels):
                if i > 3:
                    break
                instance += f"关系{i+1} 核心'{edus[inter_rel[0]]}'，依附'{edus[inter_rel[2]]}'，当前篇章关系'{inter_rel[1]}'，目标篇章关系'{rst_dct[random.choice(list(rst_dct.keys()))]}'" + "；\n"
            prompt = template +  f"输入：文本'{uttr}'\n {instance}'\n输出："
            valid_texts = []
            attempts = 0
            while len(valid_texts) < 4 and attempts < 3:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",  top_p=0.5, n=4-len(valid_texts), messages=[{"role": "user", "content": prompt}])
                texts = [completion.choices[i].message.content for i in range(len(completion.choices))]          
                # 检查文本中是否包含特定关键字
                valid_texts.extend([text[text.rindex('文本')+3:].replace('"', '').replace('\n', '【换行】') for text in texts if '文本' in text])
                attempts += 1

        elif len(uttr.split(' ')) > 3:
            single_rewrite_prompt = "请对给定文本进行适当的改写，可以改变表达方式、内容、语气，尽可能与原来不一样，如果无法进行更改，请保留原始的文本，注意：在任何时候，不要回复给定的文本"
            prompt = single_rewrite_prompt + f"给定文本：{uttr}\n改写："
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",  top_p=0.5, n=4, messages=[{"role": "user", "content": prompt}])
            texts = [completion.choices[i].message.content for i in range(len(completion.choices))]  
            valid_texts = [x.split('\n')[0] if '\n' in x else x for x in texts]
        else:
            valid_texts = [uttr for i in range(4)]
        
        for i, new_instance in enumerate(valid_texts):
            with open(f'aug/aug_uttr_free_{i}.txt', 'a', encoding='utf-8') as f:
                f.write(new_instance.replace("\'", '').replace("\"", '').strip() + '\n')


if __name__ == '__main__':
    aug_word_utterance(0)
    # aug_utterance(0)
    # aug_utterance_free(0)