import paddle
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification, BertConfig, BertModel
import pandas as pd
import glob
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
nltk.download('wordnet')
# 数据预处理
# 加载停用词列表
def load_stopwords():
    with open('stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords_list = [line.strip() for line in f.readlines()]
    return set(stopwords_list)
# 纠正错别字
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    for word in words:
        corrected_word = spell.correction(word) if spell.unknown([word]) else word
        corrected_words.append(corrected_word)
    safe_list = [item if item is not None else "" for item in corrected_words]
    return ' '.join(safe_list)
# 统一语干
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
def read_data(data_path):
    excel_files = glob.glob(data_path)
    print("所有载入表：",excel_files)
    result = []
    stopwords_set = load_stopwords()
    for file in excel_files:
        labels = file.split("\\")[-1].split(".")[0]
        df = pd.read_excel(file)
        lines = df.values.tolist()
        for row in lines:
            datas = ';'.join(str(data).strip() for data in row)
            words = datas.split()
            filtered_words = [word for word in words if word not in stopwords_set]
            filtered_text = ' '.join(filtered_words)
            # corrected_text = correct_spelling(filtered_text)
            lemmatized_text = lemmatize_text(filtered_text)
            result.append([lemmatized_text, labels])
    return result
# 锚样本选取
def select_anchor_samples(data, num_classes):
    # 向量化文本数据
    vectors = vectorize_text(data)
    # 使用 KMeans 聚类算法选取锚样本
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(vectors)
    cluster_centers = kmeans.cluster_centers_
    print("类中心：", len(cluster_centers))
    anchor_samples = []
    # 找到每个类中心最近的样本
    for i, center in enumerate(cluster_centers):
        distances = [np.linalg.norm(center - vector.toarray()[0]) for vector in vectors]
        min_distance_index = distances.index(min(distances))
        print(f"找到第{i+1}个样本：", data[min_distance_index][0])
        anchor_samples.append(data[min_distance_index])
    return anchor_samples
#mask-bert算法实现(旧)
def mask_tokens(input_ids, mlm_probability, tokenizer, mask_token_id):
    # 确保输入是PaddlePaddle的Tensor
    input_ids = paddle.to_tensor(input_ids, dtype='int64')

    # 创建一个与input_ids相同形状的概率矩阵
    prob_matrix = paddle.full(input_ids.shape, mlm_probability, dtype='float32')

    # 决定哪些标记被掩码
    masked_indices = paddle.bernoulli(prob_matrix).astype('bool')

    # 80%的掩码标记被替换为[MASK]
    random_indices = paddle.bernoulli(paddle.full(input_ids.shape, 0.8, dtype='float32')).astype(
        'bool') & masked_indices
    input_ids[random_indices] = mask_token_id

    # 10%的标记被替换为随机标记
    replacement_indices = paddle.bernoulli(paddle.full(input_ids.shape, 0.5, dtype='float32')).astype(
        'bool') & masked_indices & ~random_indices
    with paddle.no_grad():
        # 生成与replacement_indices形状相同的随机标记
        random_tokens = paddle.randint(0, len(tokenizer.vocab), paddle.sum(replacement_indices).astype('int64').shape,
                                       dtype='int64')
        # 确保赋值时形状一致
        input_ids[replacement_indices] = random_tokens

    return input_ids, masked_indices


def mask_tokens_by_contribution(input_ids, tokenizer, mlm_probability):
    # 将输入ID转换为文本
    text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # 使用TF-IDF或其他方法来评估每个词的贡献度
    tfidf_vectorizer = TfidfVectorizer()
    # 确保传递给 TfidfVectorizer 的是字符串列表
    tfidf = tfidf_vectorizer.fit_transform([text])  # text 应该是一个字符串
    feature_names = tfidf_vectorizer.get_feature_names_out()
    contributions = tfidf.toarray()[0]

    # 将贡献度分数转换为概率，较低的贡献度对应较高的掩码概率
    mask_probabilities = 1 - contributions / np.sum(contributions)

    # 确保 mask_probabilities 是一个一维数组，并且其长度与 input_ids 的长度一致
    mask_probabilities = np.array(mask_probabilities, dtype='float32')

    # 创建一个与 input_ids 相同形状的概率矩阵
    # 这里我们使用 paddle 来创建一个与 input_ids 形状相同的概率矩阵
    prob_matrix = paddle.full(input_ids.shape, mlm_probability, dtype='float32')

    # 决定哪些标记被掩码
    masked_indices = paddle.bernoulli(prob_matrix).astype('bool')

    # 将被选择的标记替换为 [MASK]
    input_ids = paddle.to_tensor(input_ids, dtype='int64')
    mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')

    # 确保只在 masked_indices 为 True 的位置上替换标记
    input_ids = paddle.where(masked_indices, paddle.full_like(input_ids, mask_token_id), input_ids)
    print("提取出来的mask掩码：",masked_indices)
    return input_ids, masked_indices
def vectorize_text(data_list):
    # 初始化 TF-IDF 向量化器
    vectorizer = TfidfVectorizer()
    # 提取所有文本数据
    texts = [item[0] for item in data_list]
    # 向量化文本数据
    vectors = vectorizer.fit_transform(texts)
    return vectors
# 数据集类
class MyDataset(Dataset):
    def __init__(self, datapath, tokenizer, num_classes, transform=None, mlm_probability=0.15, mask_token_id=None):
        super().__init__()
        self.data_list = read_data(datapath)
        self.tokenizer = tokenizer
        self.transform = transform
        self.mlm_probability = mlm_probability
        if mask_token_id is None:
            self.mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
        else:
            self.mask_token_id = mask_token_id
        self.label_to_index = {label: index for index, label in enumerate(set(item[1] for item in self.data_list))}
        # 选取锚样本
        self.anchor_samples = select_anchor_samples(self.data_list, num_classes)

    def __getitem__(self, index):
        data, label = self.data_list[index]
        is_anchor = False
        # 检查当前样本是否为锚样本
        if data in [sample[0] for sample in self.anchor_samples]:
            is_anchor = True
        encoded_inputs = self.tokenizer(
            data, max_seq_len=256, padding=True, truncation=True, return_attention_mask=True, return_tensors='pd'
        )
        input_ids = encoded_inputs['input_ids'][0]  # 获取第一个元素，因为返回的是列表
        mask_input, masked_indices = mask_tokens_by_contribution(input_ids, self.tokenizer, self.mlm_probability)
        attention_mask = encoded_inputs['attention_mask'][0]  # 同上
        label_index = self.label_to_index[label]
        return mask_input, attention_mask, label_index, masked_indices, is_anchor

    def __len__(self):
        return len(self.data_list)

# 批处理函数
def batchify_fn(batch):
    input_ids, attention_mask, labels,mask_indices,anchor_samples = zip(*batch)
    # 将 NumPy 数组转换为 PaddlePaddle 的 Tensor 列表
    sequences = [paddle.to_tensor(seq) for seq in input_ids]
    sequences1=[paddle.to_tensor(seq) for seq in attention_mask]
    # 确定最大长度
    max_len = max([len(seq) for seq in sequences])
    max_len1=max([len(seq) for seq in sequences1])
    # 填充序列
    padded_sequences = [paddle.nn.functional.pad(seq, (0, max_len - seq.shape[0]), 'constant', 0) for seq in sequences]
    padded_sequences1 = [paddle.nn.functional.pad(seq, (0, max_len - seq.shape[0]), 'constant', 0) for seq in sequences1]
    # 将填充后的序列转换为一个张量
    batch_input_ids = paddle.stack(padded_sequences)
    # batch_input_ids = paddle.to_tensor(padded_sequences, dtype='int64')
    batch_attention_mask=paddle.stack(padded_sequences1) 
    # batch_attention_mask = paddle.to_tensor(attention_mask, dtype='int64')
    batch_labels = paddle.to_tensor(labels, dtype='int64')
    return batch_input_ids, batch_attention_mask, batch_labels,mask_indices,anchor_samples

# 创建数据加载器
def create_dataloader(dataset, mode='train', batch_size=1, use_gpu=True, batchify_fn=None):
    shuffle = True if mode == 'train' else False
    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=batchify_fn, return_list=True)
    return dataloader
# 提取特征和标签
def extract_features_and_labels(dataset):
    features = []
    labels = []
    for data in dataset:
        features.append(data[0])
        labels.append(data[1])
    return features, labels
# Mask-BERT模型
class MaskBertForSequenceClassification(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(MaskBertForSequenceClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-chinese")
        self.classifier = paddle.nn.Linear(self.bert.config['hidden_size'], num_classes)

    def forward(self, input_ids, attention_mask, labels=None, masked_indices=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        print("outputs:", outputs)
        # 使用last_hidden_state代替pooler_output
        pooled_output = outputs[0]  # 或者使用outputs.last_hidden_state
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fn = paddle.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            if masked_indices is not None:
                # 计算集成梯度
                gradients = paddle.grad(loss, input_ids, create_graph=True)[0]
                attribution = paddle.sum(gradients * masked_indices.astype('float32'), axis=1)
                return loss, logits, attribution
            return loss, logits
        return logits
# 训练和评估（老）
def train_and_evaluate(model, train_loader, criterion, optimizer, epochs=20):
    global_step=0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels,label_index = batch
            # print("input_ids:",input_ids)
            # print("attention_mask:",attention_mask)
            # print("label:",labels)
            logits = model(input_ids, attention_mask=attention_mask,labels=labels)
            loss = logits.loss
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            print(f"global step {global_step}, epoch: {epoch}, loss: {loss.numpy()}")
            global_step += 1
# 定义对比损失函数
def contrastive_loss(query, key, temperature=0.07):
    logits = paddle.nn.functional.cosine_similarity(query, key) / temperature
    labels = paddle.arange(logits.shape[0], dtype=logits.dtype)
    loss = paddle.nn.functional.cross_entropy(logits, labels)
    return loss

# 定义一个函数来计算归因分数
def compute_attribution(model, input_ids, attention_mask):
    model.eval()
    gradients = paddle.grad(model(input_ids, attention_mask=attention_mask)[0], input_ids, create_graph=True)[0]
    attribution_scores = paddle.sum(gradients * input_ids, axis=-1)
    return attribution_scores


# 定义一个新的训练函数，结合对比学习和归因分数计算
def train_with_contrastive_learning(model, train_loader, epochs=5):
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-5)
    model.train()  # 确保模型处于训练模式

    for epoch in range(epochs):
        for batch in train_loader:
            input_ids, attention_mask, labels,mask_indices,is_anchor = batch
            optimizer.clear_gradients()

            # 区分锚样本和新颖样本
            if is_anchor:
                # 对锚样本使用交叉熵损失
                logits = model(input_ids, attention_mask=attention_mask)
                loss = paddle.nn.functional.cross_entropy(logits, labels)
            else:
                # 对新颖样本使用对比损失
                with paddle.no_grad():
                    # 确保锚样本的特征在计算梯度时不被更新
                    anchor_features = model(input_ids, attention_mask=attention_mask)
                query_features = model(input_ids, attention_mask=attention_mask)
                loss = contrastive_loss(query_features, anchor_features)

            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.numpy()}")

# 数据集和数据加载器
tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
# 定义掩码策略
# 数据集和数据加载器
mlm_probability = 0.15  # 15%的标记被掩码
criterion = paddle.nn.loss.CrossEntropyLoss()
# 读取数据
train_data = MyDataset('D:\\desktop\\驻场提供的数据\\*.xlsx', BertTokenizer.from_pretrained("bert-base-chinese"),num_classes=8)
print("数据集个数",len(train_data.data_list))
#创建数据集加载器
train_loader = create_dataloader(train_data, mode='train', batch_size=16, batchify_fn=batchify_fn)
# 开始训练
# model = MaskBertForSequenceClassification(num_classes=10)
model=BertForSequenceClassification.from_pretrained("bert-base-chinese", num_classes=8)
optimizer = paddle.optimizer.AdamW(learning_rate=1e-5, parameters=model.parameters(), weight_decay=0.01)
train_with_contrastive_learning(model, train_loader, epochs=5)
# 保存模型
paddle.save(model.state_dict(), 'model.pdparams')
optimizer_state_dict = optimizer.state_dict()
paddle.save(optimizer_state_dict, 'optimizer.pdopt')

