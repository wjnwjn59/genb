import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
import numpy as np
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, q):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        att = self.v_att(v, q_emb)
        att = nn.functional.softmax(att, dim=1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr
        logits = self.classifier(joint_repr)

        return logits

class GenB(nn.Module):
    def __init__(self, num_hid, dataset):
        super(GenB, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.generate = nn.Sequential(
            *block(num_hid // 8, num_hid // 4),
            *block(num_hid // 4, num_hid // 2),
            *block(num_hid // 2, num_hid),
            nn.Linear(num_hid, num_hid * 2),
            nn.ReLU(inplace=False)
        )
        self.apply(kaiming_init)

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             kaiming_init(m)

    def forward(self, v, q, gen=True):
        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        b, c, f = v.shape

        if gen:
            v_z = torch.randn(b, c, 128, device=device)
            v = self.generate(v_z.view(-1, 128)).view(b, c, f)

        att = self.v_att(v, q_emb)
        att = nn.functional.softmax(att, dim=1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr
        logits = self.classifier(joint_repr)

        return logits

class Discriminator(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dataset.num_ans_candidates, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_hid // 2),
            nn.ReLU(),
            nn.Linear(num_hid // 2, num_hid // 4),
            nn.ReLU(),
            nn.Linear(num_hid // 4, 1),
            nn.Sigmoid()
        )
        self.apply(kaiming_init)

    # def weight_init(self):
    #     for block in self._modules:
    #         for m in self._modules[block]:
    #             kaiming_init(m)

    def forward(self, z):
        return self.net(z)

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0).to(device)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0).to(device)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid).to(device)
    q_net = FCNet([num_hid, num_hid]).to(device)
    v_net = FCNet([dataset.v_dim, num_hid]).to(device)
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5
    ).to(device)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier).to(device)

def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0).to(device)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0).to(device)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid).to(device)
    q_net = FCNet([q_emb.num_hid, num_hid]).to(device)
    v_net = FCNet([dataset.v_dim, num_hid]).to(device)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5
    ).to(device)

    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier).to(device)
