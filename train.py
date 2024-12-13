import os
import time

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size(), device=device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def calc_genb_loss(logits, bias, labels):
    gen_grad = torch.clamp(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss

def train(model, genb, discriminator, train_loader, eval_loader, args, qid2type):
    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=0.001)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.001)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0

    genb.train(True)
    discriminator.train(True)

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, qid) in tqdm(enumerate(train_loader), ncols=100, desc=f"Epoch {epoch + 1}", total=len(train_loader)):
            total_step += 1

            #########################################
            v = v.clone().detach().requires_grad_(True).to(device)
            q = q.clone().detach().to(device)
            a = a.clone().detach().to(device)
            valid = torch.tensor(1.0, dtype=torch.float32).to(device).expand(v.size(0), 1)
            fake = torch.tensor(0.0, dtype=torch.float32).to(device).expand(v.size(0), 1)
            #########################################

            # get model output
            optim.zero_grad()
            pred = model(v, q)

            # train genb
            optim_G.zero_grad()
            optim_D.zero_grad()

            pred_g = genb(v, q, gen=True)
            g_loss = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_loss = g_loss * a.size(1)

            vae_preds = discriminator(pred_g)
            main_preds = discriminator(pred)

            # print(f'v: {v.shape}, q: {q.shape}, a: {a.shape}')
            # print(f'valid: {valid.shape}, fake: {fake.shape}')
            # print(f'pred_g: {pred_g.shape}, pred: {pred.shape}')
            # print(f'vae_preds: {vae_preds.shape}, main_preds: {main_preds.shape}')

            g_distill = kld(pred_g, pred.detach())
            g_dsc_loss = bce(vae_preds, valid) + bce(main_preds, valid)

            g_loss = g_loss + g_dsc_loss + g_distill * 5
            g_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)
            optim_G.step()

            # train the discriminator
            vae_preds = vae_preds.detach().requires_grad_(True).to(device)
            main_preds = main_preds.detach().requires_grad_(True).to(device)
            dsc_loss = bce(vae_preds, fake) + bce(main_preds, valid)
            dsc_loss.backward(retain_graph=True)
            optim_D.step()

            # use genb to train the robust model
            genb.train(False)
            pred_g = genb(v, q, gen=False)

            genb_loss = calc_genb_loss(pred, pred_g, a)
            genb_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            genb.train(True)
            total_loss += genb_loss.item() * q.size(0)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        logger.write(f'Epoch {epoch + 1}, time: {time.time() - t:.2f}')
        logger.write(f'\ttrain_loss: {total_loss:.2f}, score: {train_score:.2f}')

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader, qid2type)
            results["epoch"] = epoch
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results['score_yesno']
            other = results['score_other']
            num = results['score_number']
            logger.write(f'\teval score: {100 * eval_score:.2f} ({100 * bound:.2f})')
            logger.write(f'\tyn score: {100 * yn:.2f} other score: {100 * other:.2f} num score: {100 * num:.2f}')

            if eval_score > best_eval_score:
                torch.save(model.state_dict(), os.path.join(output, 'model.pth'))
                torch.save(genb.state_dict(), os.path.join(output, 'genb.pth'))
                best_eval_score = eval_score

        torch.save(model.state_dict(), os.path.join(output, 'model_final.pth'))
    print(f'best eval score: {best_eval_score * 100:.2f}')

def evaluate(model, dataloader, qid2type):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = v.clone().detach().requires_grad_(False).to(device)
        q = q.clone().detach().requires_grad_(False).to(device)
        pred = model(v, q)
        batch_score = compute_score_with_logits(pred, a.to(device)).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        qids = qids.detach().cpu().int().numpy()
        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    results = dict(
        score=score,
        upper_bound=upper_bound,
        score_yesno=score_yesno,
        score_other=score_other,
        score_number=score_number,
    )
    return results
