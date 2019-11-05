import torch
import torch.nn as nn

def Lsup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return (loss1 + loss2)


def Lcot(U_p1, U_p2):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)
    U_batch_size = U_p1.size()[0]

    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)
    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


def Ldiff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2, logit_U1, logit_U2, perturbed_logit_U1,
          perturbed_logit_U2):
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)
    batch_size = logit_S1.size()[0] + logit_U1.size()[0]

    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    return -(a + b + c + d) / batch_size

