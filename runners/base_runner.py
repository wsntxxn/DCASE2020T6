import os
import sys
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
from tqdm import tqdm
import fire
import torch
from ignite.engine.engine import Engine
from ignite.contrib.handlers import ProgressBar

sys.path.append(os.getcwd())
import utils.train_util as train_util
from datasets.SJTUDataSet import SJTUDataset, SJTUDatasetEval, collate_fn

class BaseRunner(object):
    """Main class to run experiments"""
    def __init__(self, seed=1):
        super(BaseRunner, self).__init__()
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = torch.device(device)

    @staticmethod
    def _get_dataloaders(config, vocabulary):
        scaler = getattr(
            pre, config["scaler"])(
            **config["scaler_args"])
        inputdim = -1
        caption_df = pd.read_json(config["caption_file"], dtype={"key": str})

        for batch in tqdm(
            torch.utils.data.DataLoader(
                SJTUDataset(
                    feature=config["feature_file"],
                    caption_df=caption_df,
                    vocabulary=vocabulary,
                ),
                collate_fn=collate_fn([0, 1]),
                **config["dataloader_args"]
            ),
            ascii=True
        ):
            feat = batch[0]
            feat_lens = batch[-2]
            packed_feat = torch.nn.utils.rnn.pack_padded_sequence(
                feat, feat_lens, batch_first=True, enforce_sorted=False).data
            scaler.partial_fit(packed_feat)
            inputdim = feat.shape[-1]
        assert inputdim > 0, "Reading input feature failed"

        augments = train_util.parse_augments(config["augments"])
        train_keys = np.random.choice(
            caption_df["key"].unique(), 
            int(len(caption_df["key"].unique()) * (config["train_percent"] / 100.)), 
            replace=False
        )
        train_df = caption_df[caption_df["key"].apply(lambda x: x in train_keys)]
        val_df = caption_df[~caption_df.index.isin(train_df.index)]

        train_loader = torch.utils.data.DataLoader(
            SJTUDataset(
                feature=config["feature_file"],
                caption_df=train_df,
                vocabulary=vocabulary,
                transform=[scaler.transform, augments]
            ),
            shuffle=True,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"]
        )

        if config["zh"]:
            train_key2refs = train_df.groupby("key")["tokens"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            train_key2refs = train_df.groupby("key")["caption"].apply(list).to_dict()
            val_key2refs = val_df.groupby("key")["caption"].apply(list).to_dict()
        val_loader = torch.utils.data.DataLoader(
            SJTUDataset(
                feature=config["feature_file"],
                caption_df=val_df,
                vocabulary=vocabulary,
                transform=scaler.transform
            ),
            shuffle=False,
            collate_fn=collate_fn([0, 1]),
            **config["dataloader_args"])

        return train_loader, val_loader, {
            "scaler": scaler, "inputdim": inputdim, 
            "train_key2refs": train_key2refs, "val_key2refs": val_key2refs
        }

    @staticmethod
    def _get_model(config, vocabulary):
        raise NotImplementedError

    def _forward(self, model, batch, mode, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _convert_idx2sentence(word_ids, vocabulary, zh=False):
        candidate = []
        for word_id in word_ids:
            word = vocabulary.idx2word[word_id]
            if word == "<end>":
                break
            elif word == "<start>":
                continue
            candidate.append(word)
        if not zh:
            candidate = " ".join(candidate)
        return candidate

    def train(self, config, **kwargs):
        raise NotImplementedError

    def sample(self,
               experiment_path: str,
               feature_file: str,
               feature_scp: str,
               output: str="output_word.txt",
               **kwargs):
        """Generate captions given experiment model"""
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""
        import tableprint as tp

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        # Load previous training config
        config = dump["config"]

        vocab_size = len(torch.load(config["vocab_file"]))
        model = self._get_model(config, vocab_size)
        model.load_state_dict(dump["model"])
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        vocabulary = torch.load(config["vocab_file"])
        zh = config["zh"]
        model = model.to(self.device)
        dataset = SJTUDatasetEval(
            feature=feature_file,
            eval_scp=feature_scp,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=16,
            num_workers=0)

        width_length = 80
        pbar = ProgressBar(persist=False, ascii=True)
        writer = open(os.path.join(experiment_path, output), "w")
        writer.write(
            tp.header(
                ["InputUtterance", "Output Sentence"], width=[len("InputUtterance"), width_length]))
        writer.write('\n')

        sentences = []
        def _sample(engine, batch):
            # batch: [keys, feats, feat_lens]
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="sample", **kwargs)
                seqs = output["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh=zh)
                    if zh:
                        sentence = " ".join(caption)
                    else:
                        sentence = caption
                    writer.write(tp.row([keys[idx], sentence], width=[len("InputUtterance"), width_length]) + "\n")
                    sentences.append(sentence)

        sample_engine = Engine(_sample)
        pbar.attach(sample_engine)
        sample_engine.run(dataloader)
        writer.write(tp.bottom(2, width=[len("InputUtterance"), width_length]) + "\n")
        writer.write("Unique sentence number: {}\n".format(len(set(sentences))))
        writer.close()

    def evaluate(self,
                 experiment_path: str,
                 feature_file: str,
                 feature_scp: str,
                 caption_file: str,
                 caption_output: str = "eval_output.json",
                 score_output: str = "scores.txt",
                 **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        # Load previous training config
        config = dump["config"]

        vocabulary = torch.load(config["vocab_file"])
        model = self._get_model(config, vocabulary)
        model.load_state_dict(dump["model"])
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        zh = config["zh"]
        model = model.to(self.device)

        dataset = SJTUDatasetEval(
            feature=feature_file,
            eval_scp=feature_scp,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0)

        caption_df = pd.read_json(caption_file, dtype={"key": str})
        if zh:
            key2refs = caption_df.groupby("key")["tokens"].apply(list).to_dict()
        else:
            key2refs = caption_df.groupby("key")["caption"].apply(list).to_dict()

        model.eval()

        key2pred = {}

        def _sample(engine, batch):
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="sample", **kwargs)
                seqs = output["seqs"].cpu().numpy()

                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh)
                    key2pred[keys[idx]] = [caption,]

        pbar = ProgressBar(persist=False, ascii=True)
        sampler = Engine(_sample)
        pbar.attach(sampler)
        sampler.run(dataloader)

        pred_df = []
        for key, pred in key2pred.items():
            pred_df.append({
                "filename": key + ".wav",
                "caption": "".join(pred[0]) if zh else pred[0],
                "tokens": pred[0] if zh else pred[0].split() 
            })
        pred_df = pd.DataFrame(pred_df)
        pred_df.to_json(os.path.join(experiment_path, caption_output))

        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.spice.spice import Spice

        f = open(os.path.join(experiment_path, score_output), "w")

        scorer = Bleu(n=4, zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        for n in range(4):
            f.write("Bleu-{}: {:6.3f}\n".format(n + 1, score[n]))

        scorer = Rouge(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("ROUGE: {:6.3f}\n".format(score))

        scorer = Cider(zh=zh)
        score, scores = scorer.compute_score(key2refs, key2pred)
        f.write("CIDEr: {:6.3f}\n".format(score))

        if not zh:
            scorer = Meteor()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Meteor: {:6.3f}\n".format(score))

            scorer = Spice()
            score, scores = scorer.compute_score(key2refs, key2pred)
            f.write("Spice: {:6.3f}\n".format(score))

        f.close()

    def dcase_predict(self,
                      experiment_path: str,
                      feature_file: str,
                      predict_scp: str,
                      output: str="predition.csv",
                      **kwargs):
        """kwargs: {'max_length': int, 'method': str, 'beam_size': int}"""

        dump = torch.load(os.path.join(experiment_path, "saved.pth"),
                          map_location="cpu")
        # Load previous training config
        config = dump["config"]

        vocabulary = torch.load(config["vocab_file"])
        model = self._get_model(config, len(vocabulary))
        model.load_state_dict(dump["model"])
        # Some scaler (sklearn standardscaler)
        scaler = dump["scaler"]
        zh = config["zh"]
        model = model.to(self.device)

        dataset = SJTUDatasetEval(
            feature=feature_file,
            eval_scp=predict_scp,
            transform=scaler.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn((1,)),
            batch_size=32,
            num_workers=0)

        pbar = ProgressBar(persist=False, ascii=True)
        predictions = []

        def _sample(engine, batch):
            # batch: [keys, feats, feat_lens]
            with torch.no_grad():
                model.eval()
                keys = batch[0]
                output = self._forward(model, batch, mode="sample", **kwargs)
                seqs = output["seqs"].cpu().numpy()
                for idx, seq in enumerate(seqs):
                    caption = self._convert_idx2sentence(seq, vocabulary, zh=zh)
                    predictions.append({"file_name": keys[idx] + ".wav", "caption_predicted": caption})

        sample_engine = Engine(_sample)
        pbar.attach(sample_engine)
        sample_engine.run(dataloader)

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(experiment_path, output), index=False)

if __name__ == "__main__":
    fire.Fire(BaseRunner)
