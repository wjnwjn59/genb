import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from dataset import Dictionary, VQAFeatureDataset
import base_model
from tqdm import tqdm

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, dim=1)
    one_hots = torch.zeros_like(labels, device=device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels).sum(dim=1)
    return scores

def evaluate(model, dataloader, qid2type):
    score, upper_bound = 0, 0
    score_yesno, score_number, score_other = 0, 0, 0
    total_yesno, total_number, total_other = 0, 0, 0

    for v, q, a, qids in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v, q, a = v.to(device), q.to(device), a.to(device)
        pred = model(v, q)

        batch_score = compute_score_with_logits(pred, a).cpu().numpy()
        score += batch_score.sum()
        upper_bound += a.max(dim=1)[0].sum().item()

        qids = qids.cpu().numpy()
        for idx, qid in enumerate(qids):
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[idx]
                total_yesno += 1
            elif typ == 'number':
                score_number += batch_score[idx]
                total_number += 1
            elif typ == 'other':
                score_other += batch_score[idx]
                total_other += 1

    score /= len(dataloader.dataset)
    upper_bound /= len(dataloader.dataset)
    score_yesno = score_yesno / total_yesno if total_yesno > 0 else 0
    score_number = score_number / total_number if total_number > 0 else 0
    score_other = score_other / total_other if total_other > 0 else 0

    return score, upper_bound, score_yesno, score_number, score_other

def parse_args():
    parser = argparse.ArgumentParser("Evaluate the BottomUpTopDown model with a de-biasing method")

    parser.add_argument('--cache_features', action='store_true', help="Cache image features in RAM for faster evaluation")
    parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1"], help="Dataset selection")
    parser.add_argument('--num_hid', type=int, default=1024, help="Number of hidden units")
    parser.add_argument('--model', type=str, default='baseline0_newatt', help="Model type")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--seed', type=int, default=1111, help="Random seed")
    parser.add_argument('--load_path', type=str, default='best_model', help="Path to the trained model")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset and dictionary
    dictionary_path = 'data/dictionary.pkl' if args.dataset in ['cpv2', 'v2'] else 'data/dictionary_v1.pkl'
    dictionary = Dictionary.load_from_file(dictionary_path)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary, dataset=args.dataset, cache_image_features=args.cache_features)

    # Build the model
    constructor = f'build_{args.model}'
    model = getattr(base_model, constructor)(eval_dset, args.num_hid).to(device)

    # Load pre-trained model
    model_path = os.path.join(args.load_path, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded Model!")

    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Load qid2type mapping
    with open(f'util/qid2type_{args.dataset}.json', 'r') as f:
        qid2type = json.load(f)

    # Evaluation
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    eval_score, bound, yn, num, other = evaluate(model, eval_loader, qid2type)

    print(f'\teval score: {eval_score * 100:.2f} ({bound * 100:.2f})')
    print(f'\tyn score: {yn * 100:.2f}, number score: {num * 100:.2f}, other score: {other * 100:.2f}')

if __name__ == '__main__':
    main()
