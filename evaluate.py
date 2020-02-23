import torch
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from args import parser
from utils import *

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load(args.model_dir)
model.to(device)

data = load_pt(args.test_data)
dataloader = DataLoader(data, batch_size=args.batch_size)

label2idx = load_pt('data/label2idx.pt')
idx2label = {idx: label for label, idx in label2idx.items()}

result = open(args.out_file, 'w')
result_writer = csv.writer(result)
result_writer.writerow(['id', 'label'])

with torch.no_grad():
    for batch in tqdm(dataloader):
        idx, x = batch
        x = x.to(device)
        pred = model(x)

        pred_id = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
        for i, label in zip(idx, pred_id):
            result_writer.writerow([i, idx2label[label]])

result.close()
