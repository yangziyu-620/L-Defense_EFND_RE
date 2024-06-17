# data process
# LIAR-RAW
# RAWFC
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
from os.path import join as pjoin
from help import *

HOME_DIR = BASE_DIR + "/L-Defense_EFND_RE/"

DATASET2PATH = {
    "LIAR_RAW": os.path.join(HOME_DIR + "/dataset/LIAR-RAW/"),
    "RAWFC": os.path.join(HOME_DIR + "/dataset/RAWFC/"),

    "RAWFC_step2": os.path.join(HOME_DIR + "/dataset/RAWFC_step2/"),
    "LIAR-RAW_step2": os.path.join(HOME_DIR + "/dataset/LIAR-RAW_step2/"),

}
# print(DATASET2PATH)

LABEL_IDS = {
    "LIAR_RAW": {"pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 1, "mostly-true": 2, "true": 2},
    "RAWFC": {"false": 0, "half": 1, "true": 2},
    "LIAR_RAW_SIX": {"pants-fire": 0, "false": 1, "barely-true": 2, "half-true": 3, "mostly-true": 4, "true": 5},
}


def get_LIAR_six_cls_labels(dataset_type):
    train_dataset_raw, dev_dataset_raw, test_dataset_raw = get_raw_datasets("LIAR_RAW")
    labels = {}
    if dataset_type == 'train':
        dataset = train_dataset_raw
    elif dataset_type == 'eval':
        dataset = dev_dataset_raw
    elif dataset_type == 'test':
        dataset = test_dataset_raw

    for obj in tqdm(dataset):
        # each sample has one
        event_id, label = obj['event_id'], obj['label']
        labels[event_id] = LABEL_IDS["LIAR_RAW_SIX"][label]

    return labels


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        news_dataset = json.load(f)
    return news_dataset


def read_RAWFC(path):
    filenames = os.listdir(path)
    name_list = []
    for name in filenames:
        if '.json' in name:
            name_list.append(name)

    if len(name_list) == 1:
        all_data = ''
        for file in name_list:
            filename = pjoin(path, file)
            with open(filename, 'r', encoding='utf-8') as json_file:
                all_data = json.load(json_file)
    else:
        all_data = []
        for file in name_list:
            filename = pjoin(path, file)
            with open(filename, 'r', encoding='utf-8') as json_file:
                obj = json.load(json_file)
                all_data.append(obj)
    return all_data


def get_raw_datasets(dataset, dataset_dir=None):
    if dataset == "LIAR_RAW":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train.json")
        dev_dataset_path = os.path.join(dataset_dir, "val.json")
        test_dataset_path = os.path.join(dataset_dir, "test.json")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset == 'RAWFC':
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train")
        dev_dataset_path = os.path.join(dataset_dir, "val")
        test_dataset_path = os.path.join(dataset_dir, "test")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_RAWFC(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]
    elif dataset == "RAWFC_step2" or dataset == "LIAR_RAW_step2":
        dataset_dir = dataset_dir or DATASET2PATH[dataset]
        train_dataset_path = os.path.join(dataset_dir, "train_10_evidence_details.json")
        # if top-k == 10
        dev_dataset_path = os.path.join(dataset_dir, "eval_10_evidence_details.json")
        test_dataset_path = os.path.join(dataset_dir, "test_10_evidence_details.json")
        train_dataset_raw, dev_dataset_raw, test_dataset_raw = [
            read_json(_p) for _p in [train_dataset_path, dev_dataset_path, test_dataset_path]]

    else:
        raise NotImplementedError(dataset, dataset_dir)
    return train_dataset_raw, dev_dataset_raw, test_dataset_raw


class NewsDataset(Dataset):
    def __init__(
            self, dataset_name, news_dataset, tokenizer, max_seq_length=128,
            nums_label=6, report_each_claim=None, *args, **kwargs
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.nums_label = nums_label
        self.label_dir = LABEL_IDS[dataset_name]
        self.dataset_path = None
        self.report_each_claim = report_each_claim

        self.example_list = news_dataset
        self.event_id, self.claim, self.label, self.explain, \
            self.report_links, self.report_contents, self.report_domains, \
            self.report_sents, self.report_sents_labels, \
            self.report_sents_belong, self.num_sentences_per_report = self.load_raw(news_dataset)

        self._sep_id, self._cls_id, self._pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token, self.tokenizer.cls_token, self.tokenizer.pad_token, ]
        )

    def load_raw(self, dataset):
        """parsing dict objs to list """
        # event_id, claim, label, explain,
        # (link, content, domain, report_sents, report_sents_is_evidence,
        # report_sents_belong_which_report)

        raw_data = [[] for _ in range(11)]
        for obj in tqdm(dataset):
            # each sample has one
            event_id, claim, label, explain, reports = \
                obj['event_id'], obj['claim'], obj['label'], obj['explain'], obj['reports']
            raw_data[0].append(event_id)
            raw_data[1].append(claim)
            raw_data[2].append(label)
            raw_data[3].append(explain)

            # each sample has many reports
            report_links = []
            report_contents = []
            report_domains = []

            report_sents = []
            report_sents_labels = []
            report_sents_belong = []

            num_sentences_per_report = []

            for r_id, report in enumerate(reports[:self.report_each_claim]):  # clip
                report_links.append(report['link'])
                report_contents.append(report['content'])  # complete report
                report_domains.append(report['domain'])
                num_sentences_per_report.append(len(report['tokenized']))
                # each report has many sentences
                for sentence in report['tokenized']:
                    report_sents.append(sentence['sent'])
                    report_sents_labels.append(sentence['is_evidence'])
                    report_sents_belong.append(r_id)

                raw_data[4].append(report_links)
                raw_data[5].append(report_contents)
                raw_data[6].append(report_domains)

                raw_data[7].append(report_sents)
                raw_data[8].append(report_sents_labels)
                raw_data[9].append(report_sents_belong)

                raw_data[10].append(num_sentences_per_report)

            return raw_data

