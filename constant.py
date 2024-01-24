
rel_dct = {
    'root': '根节点',
    'sasubj-obj': '同主同宾',
    'sasubj': '同主语',
    'dfsubj': '不同主语',
    'subj': '主语',
    'subj-in': '内部主语',
    'obj': '宾语',
    'pred': '谓语',
    'att': '定语',
    'adv': '状语',
    'cmp': '补语',
    'coo': '并列',
    'pobj': '介宾',
    'iobj': '间宾',
    'de': '的',
    'adjct': '附加',
    'app': '称呼',
    'exp': '解释',
    'punc': '标点',
    'frag': '片段',
    'repet': '重复',
    # rst
    'attr': '归属',
    'bckg': '背景',
    'cause': '因果',
    'comp': '比较',
    'cond': '状况',
    'cont': '对比',
    'elbr': '阐述',
    'enbm': '目的',
    'eval': '评价',
    'expl': '解释-例证',
    'joint': '联合',
    'manner': '方式',
    'rstm': '重申',
    'temp': '时序',
    'tp-chg': '主题变更',
    'prob-sol': '问题-解决',
    'qst-ans': '疑问-回答',
    'stm-rsp': '陈述-回应',
    'req-proc': '需求-处理',
}

relsyn2id = {}
for i, (key, value) in enumerate(rel_dct.items()):
    if i <= 20:  # only syntax
        relsyn2id[key] = i

rel2id = {}
for i, (key, value) in enumerate(rel_dct.items()):
        rel2id[key] = i

id2rel = list(rel_dct.keys())

punct_lst = ['，', '.', '。', '!', '?', '~', '...', '......', ',', ':', '：', ';']    

name2rel = {
    'attribution':'attr',
    'background':'bckg',
    'cause':'cause',
    'comparison':'comp',
    'condition':'cond',
    'contrast':'cont',
    'elaboration':'elbr',
    'enablement':'enbm',
    'evaluation':'eval',
    'explanation':'expl',
    'joint':'joint',
    'manner-means':'manner',
    'restatement':'rstm',
    'temporal':'temp',
    'topic-change':'tp-chg',
    'problem-solution':'prob-sol',
    'question-answer':'qst-ans',
    'statement-response':'stm-rsp',
    'request-process':'req-proc',
}