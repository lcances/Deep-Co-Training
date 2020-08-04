import torch
import torch.nn as nn

def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return (loss1 + loss2)


def p_loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return loss1, loss2, (loss1 + loss2)


def loss_cot(U_p1, U_p2):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)
    U_batch_size = U_p1.size()[0]
    eps=1e-8

    a1 = 0.5 * (S(U_p1) + S(U_p2))
    a1 = torch.clamp(a1, min=eps)
    
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)

    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)

    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / U_batch_size


def loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2,
              logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)

    S_batch_size = logit_S1.size()[0]
    U_batch_size = logit_U1.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    return -(a + b + c + d) / total_batch_size


def p_loss_diff(logit_S1, logit_S2, perturbed_logit_S1, perturbed_logit_S2,
                logit_U1, logit_U2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)

    S_batch_size = logit_S1.size()[0]
    U_batch_size = logit_U1.size()[0]
    total_batch_size = S_batch_size + U_batch_size

    a = S(logit_S2) * LS(perturbed_logit_S1)
    a = torch.sum(a)

    b = S(logit_S1) * LS(perturbed_logit_S2)
    b = torch.sum(b)

    c = S(logit_U2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(logit_U1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    pld_S = -(a + b) / S_batch_size
    pld_U = -(c + d) / U_batch_size
    ldiff = -(a + b + c + d) / total_batch_size

    return pld_S, pld_U, ldiff
