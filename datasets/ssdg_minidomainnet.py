import os.path as osp
import glob
import random

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .ssdg_pacs import SSDGPACS


@DATASET_REGISTRY.register()
class SSDGMiniDomainNet(DatasetBase):
    """A subset of DomainNet.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning.
    """

    dataset_dir = "domainnet"
    domains = ["clipart", "painting", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_ssdg_dir = osp.join(self.dataset_dir, "splits_mini_ssdg")
        mkdir_if_missing(self.split_ssdg_dir)
        
        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        seed = cfg.SEED
        num_labeled = cfg.DATASET.NUM_LABELED
        src_domains = cfg.DATASET.SOURCE_DOMAINS
        tgt_domain = cfg.DATASET.TARGET_DOMAINS[0]
        split_ssdg_path = osp.join(
            self.split_ssdg_dir, f"{tgt_domain}_nlab{num_labeled}_seed{seed}.json"
        )

        if not osp.exists(split_ssdg_path):
            train_x, train_u = self._read_data_train(
                cfg.DATASET.SOURCE_DOMAINS, "train", num_labeled
            )
            SSDGPACS.write_json_train(
                split_ssdg_path, src_domains, self.dataset_dir, train_x, train_u
            )
        else:
            train_x, train_u = SSDGPACS.read_json_train(
                split_ssdg_path, src_domains, self.dataset_dir
            )
        val = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, "val")
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, "all")

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x
        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def _read_data_train(self, input_domains, split, num_labeled):
        items_x, items_u = [], []
        num_labeled_per_class = None
        num_domains = len(input_domains)

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_ssdg_dir, filename)
            datas = dict()
            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    if int(label) not in datas.keys():
                        datas[int(label)] = [impath]
                    else:
                        datas[int(label)].append(impath)
            
            if num_labeled_per_class is None:
                num_labeled_per_class = num_labeled / (num_domains * len(datas.keys()))

            for label, impaths in datas.items():
                #assert len(impaths) >= num_labeled_per_class
                random.shuffle(impaths)

                for i, impath in enumerate(impaths):
                    item = Datum(impath=impath, label=label, domain=domain)
                    if (i + 1) <= num_labeled_per_class:
                        items_x.append(item)
                    else:
                        items_u.append(item)
        # # of total train data : 119413
        return items_x, items_u

    def _read_data_test(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                filename = dname + "_" + "train" + ".txt"
                train_dir = osp.join(self.split_ssdg_dir, filename)
                impath_label_list = self._read_split_domainnet(train_dir)
                
                filename = dname + "_" + "test" + ".txt"
                val_dir = osp.join(self.split_ssdg_dir, filename)
                impath_label_list += self._read_split_domainnet(val_dir)
            else:
                if split == "val":
                    filename = dname + "_" + "test" + ".txt"
                else:
                    filename = dname + "_" + split + ".txt"
                split_dir = osp.join(self.split_ssdg_dir, filename)
                impath_label_list = self._read_split_domainnet(split_dir)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items

    def _read_split_domainnet(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                impath = osp.join(self.dataset_dir, impath)
                label = int(label)
                items.append((impath, label))

        return items