## Tianchi Ncov Text Similarity Competition

#### Note: 
- pretrained_bert_model need contains pretrained bert model. You need download from [here](https://drive.google.com/open?id=1il88pC5DabgypSYAF8pq_E2cuNrNuUAC) and unzip the file into pretrained_bert_model folder.

- In my experiment, I use [**ERNIE 1.0 Base for Chinese(pre-train step max-seq-len-512)**](https://github.com/nghuyong/ERNIE-Pytorch) as pretrained model, you can choose **BERT**.(I havn't test the acc using bert.)

### ACC 0.93-0.94
```bash
python prepare.py --bert
python train.py --bert --drop_out 0.3 --d_hidden 1024 --batch_size 32 --lr 1e-4
```