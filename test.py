import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=10)
optimizer = paddle.optimizer.AdamW(learning_rate=1e-5, parameters=model.parameters(), weight_decay=0.01)

# 加载模型参数
model_state_dict = paddle.load('model.pdparams')
model.set_dict(model_state_dict)

# 加载优化器状态
optimizer_state_dict = paddle.load('optimizer.pdopt')
optimizer.set_state_dict(optimizer_state_dict)


def batchify_fn(samples):
    # 假设 samples 是一个列表，其中每个元素都是一个包含 input_ids 和 attention_mask 的元组
    input_ids, attention_masks = zip(*samples)
    print("input_ids:",len(input_ids[0]),len(input_ids[1]))
    # 找到最长的序列长度
    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids=paddle.to_tensor(input_ids)
    padded_attention_masks=paddle.to_tensor(attention_masks)
    # # 将序列转换为 Tensor 并填充
    # padded_input_ids = paddle.stack([
    #     paddle.nn.functional.pad(paddle.to_tensor(ids), [0, 0, 0, max_len - ids.shape[0]], 'constant', value=tokenizer.pad_token_id)
    #     for ids in input_ids
    # ])
    # padded_attention_masks = paddle.stack([
    #     paddle.nn.functional.pad(paddle.to_tensor(mask), [0, 0, 0, max_len - mask.shape[0]], 'constant', value=0)
    #     for mask in attention_masks
    # ])

    return (padded_input_ids, padded_attention_masks)

def convert_example(example, tokenizer, max_length=256, is_test=False):
    text = example
    encoded_inputs = tokenizer(
        text, max_seq_len=max_length, padding=True, truncation=True, return_attention_mask=True,
        return_tensors='pd'  # 修改这里
    )
    input_ids = encoded_inputs["input_ids"][0]
    attention_mask = encoded_inputs["attention_mask"][0]
    return input_ids, attention_mask

def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    for text in data:
        input_ids, attention_mask = convert_example(text, tokenizer, is_test=True)
        examples.append((input_ids, attention_mask))

    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, attention_mask = batchify_fn(batch)
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results
data = ['0001f912a3e14d5b83ac5dc0d9469921;GKGXM156000000809259W;220100.0;220000;220100.0;nan;国家自然科学基金_网采;nan;国家自然科学基金;nan;nan;10.0;11901056;动力系统中旋转周期解Hopf分支问题研究;nan;nan;TP;nan;2019;nan;2020-01;nan;2022-12;nan;nan;nan;nan;nan;TP271.81;自动化系统;4131015;指挥与控制系统工程;nan;nan;nan;nan;nan;nan;22.0;nan;nan;nan;nan;nan;nan;nan;nan;0;00:00:00;nan;nan;nan;nan;1.0;nan;0.0;nan;nan;nan;nan;bbb04b0e-452c-11e8-8489-0050569c;nan;nan;nan;nan;nan;nan;其他;nan;nan;GKGRY156000002314520R;王帅;GKGJG156000000110202D;长春理工大学;nan;nan;nan;nan;十三五;nan;1;nan;2.0;412756090;nan;nan;王帅;1.0;1987-02-20;1.0;4.1018319870220006e+17;长春理工大学;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;东北;2.0;nan;nan;6ce1e6978a7f4b1e8f572fc437443575;2021-04-21 11:06:49;2022-10-11 14:13:44;0;nan',
        '9c79589aa97911ebbb39b02628475330;4f6df4b53ce843fb82523cc8e24906b2;nan;nan;nan;GKGCC0320160049B2384W;44316.4512962963;nan;nan;Y3541010;做市商制度对我国新三板市场收益率和流动性影响的实证分析;杨震寰;应用统计;邵新慧;2;东北大学;nan;F832.51;证券市场;做市商制度;收益率;流动性;做市商制度由于其理论上能够改善股票流动性、稳定收益率，并且在国外的证券市场上良好的实际表现，于2014年8月25日引入到我国的新三板市场中。由于我国的证券市场发展不如发达国家成熟，很多相关政策并未规范，并且新三板市场中均为中小微公司，做市商制度是否确实达到上述预期目标，是一个值得研究的课题。<br>　　本文首先总结了做市商制度的基本特征，系统阐述了国内外关于做市商制度的理论研究和实践经验。其次，本文通过对证券市场中流动性的描述，提取出合理的流动性特征，并结合实际情况，建立了符合研究需求的流动性指标。再次，为了探究做市商制度对收益率和流动性之间的影响，本文采取了VAR建模、格兰杰因果检验、脉冲响应函数分析和方差分解的研究方法。最后，在实证部分，本文采用了2015年12月至2016年9月的新三板实际数据作为样本。通过以上建模研究方法，并且把样本分成三个期间，运用横向和纵向比较，对做市转让方式和非做市转让方式之间的差异进行分析。<br>　　分析结果显示，做市商制度起到了稳定收益率、增大流动性的作用，但是在经济持续下行的环境中，流动性反而大幅下降;收益率影响流动性，与非做市转让方式相反;收益率对流动性的冲击影响逐渐加强，但总体上仍然偏弱。综合以上几点分析结果，可以认为:我国新三板市场中的做市商制度在活跃市场上发挥出了一定的作用，比原来的制度有了一定的提升;但是作用还不全面，未能完全达到预期，并且作用发挥不够稳定，容易受到市场行情的影响。;nan;nan;nan;nan;2016;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;0;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;0;0;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;2021-04-30;1638243867729;1638243867729;nan;学位论文（万方）_20210420102830;nan;nan',
        '9cb6a170a97911ebbb39b02628475330;4f6df4b53ce843fb82523cc8e24906b2;nan;nan;nan;GKGCC0320190049B494ED;44316.4512962963;nan;nan;Y3556967;作业成本法在L营养品公司的应用研究;李香慧;会计硕士;黄莉;2;西安石油大学;nan;F407.82;营养品公司;成本管理;作业成本法;市场竞争;近年来由于政策的鼓励以及市场的热情，营养品行业发展迅速，企业竞争也愈加激烈。L营养品公司尽管起步较晚，规模较小，但正处在高速发展阶段，公司产品种类丰富，生产工艺也各不相同，使用的生产设备高端智能，但目前使用的成本核算方法对于制造费用分配并不准确，这导致企业目前的成本信息并不真实。而作业成本法通过对制造费用的精准分配，可以使企业获得更为准确的产品成本，进而进行更恰当的经营决策。<br>　　本文以L营养品公司为研究对象，对作业成本法在营养品行业的应用进行了初步研究。首先针对L营养品公司目前的成本核算与管理现状进行分析，探讨存在的局限性，以及引入作业成本法的必要性和可行性。然后通过梳理L营养品公司的作业流程，根据其不同产品的生产工艺，确定各项作业，合理划分作业中心，确定相关成本动因，并以此为基础完成制造费用的分配，设立了完整的作业成本法应用方案。结合企业具体成本数据，分别以该方案和传统成本法进行核算，比较二者之间的差异，并对该差异进行分析。最后探讨了作业成本法L营养品公司进行成本控制、产品定价以及产品结构方面的应用效果，并提出了相关保证措施。本文的研究发现了作业成本法在营养品行业应用的可行性，其应用过程为整个营养品制造行业应用作业成本法提供了参考。;nan;nan;nan;nan;2019;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;0;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;0;0;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;nan;2021-04-30;1638243889187;1638243889187;nan;学位论文（万方）_20210420102830;nan;nan']
label_map = {0: 'EI论文', 1: 'SCI论文',2:'工作簿1',3:'会议论文',4:'机构数据',5:'期刊论文',6:'学位论文',7:'自科基金项目数据'}

predictions = predict(model, data, tokenizer, label_map, batch_size=32)
for idx, text in enumerate(data):
    print('预测文本: {} \n标签: {}'.format(text, predictions[idx]))
