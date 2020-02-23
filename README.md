## Tianchi Ncov Text Similarity Competition

#### Note: pretrained_bert_model need contains pretrained bert model.
### Step 1:
Prepare the structured data
``` bash
python prepare.py --esim
```
### Step 2:
Begin train
```bash
python train.py --esim --drop_out 0.5 --embedding_dim 256 --d_hidden 1024 --batch_size 64
```