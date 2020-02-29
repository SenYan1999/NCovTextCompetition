import torch
import csv
from tqdm import tqdm
from torch.utils.data import DataLoader
from args import parser
from utils import BiSentDataset

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = torch.load(args.model_dir)
model.to(device)

# prepare test data
data = BiSentDataset(args.raw_test_data, args.max_len, test=True)
dataloader = DataLoader(data, batch_size=args.batch_size)

result = open(args.out_file, 'w')
result_writer = csv.writer(result)
result_writer.writerow(['id', 'label'])

with torch.no_grad():
    for batch in tqdm(dataloader):
        idx, x, y, input_id = map(lambda x: x.to(device), batch)
        pred = model(x, input_id)

        pred_id = torch.argmax(pred, dim=-1).cpu().numpy().tolist()
        for i, label in zip(idx, pred_id):
            result_writer.writerow([i.item(), label])

result.close()