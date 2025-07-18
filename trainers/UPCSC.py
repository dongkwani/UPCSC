import contextlib
import random
import os
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param

from .adain.adain import AdaIN
import copy


@contextlib.contextmanager
def freeze_models_params(models):
    try:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(False)
        yield
    finally:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(True)


class adaptiveUPCSCLoss(nn.Module):
    def __init__(self, cfg, num_classes, scale=1):
        super().__init__()
        self.cfg = cfg
        self.soft_plus = nn.Softplus()
        self.scale = scale
        self.num_classes = num_classes
        self.low_thre = 1/num_classes


    def forward(self, feature, cP, C, prob_detach, conf_mask): 
        feature = F.normalize(feature, p=2, dim=1)
        proxy = C.get_proxy()    # (C, fdim) or (C, fdim+1)
        proxy = F.normalize(cP(proxy), p=2.0, dim=1)   # (C,dim)

        class_candidates = (prob_detach > self.low_thre)  # (B,C)

        _, top1_idx = torch.max(prob_detach, dim=1)
        top1_class = torch.zeros_like(prob_detach, dtype=torch.bool).to(prob_detach.device)
        top1_class[torch.arange(top1_class.shape[0]), top1_idx] = True

        pred_class = class_candidates * ~conf_mask.unsqueeze(1) + top1_class * conf_mask.unsqueeze(1)

        sc_weight = prob_detach * class_candidates
        sc_proxy = (sc_weight.unsqueeze(2) * proxy).sum(dim=1)  # (B,dim)
        sc_sim = (feature * sc_proxy).sum(dim=1)  # (B,)

        p_maxval, target = prob_detach.max(1)
        pred = torch.matmul(feature, proxy.T)
        proxy_sim = pred[torch.arange(feature.shape[0]), target]    # (B,)
        pos_pair = proxy_sim * conf_mask + sc_sim * ~conf_mask

        pred_class_1 = pred_class.unsqueeze(1)  # (B, 1, C)
        pred_class_2 = pred_class.unsqueeze(0)  # (1, B, C)
        intersection = (pred_class_1 & pred_class_2).any(dim=2).bool()  # (B,B)
        neg_matrix = ~intersection

        feature = torch.matmul(feature, feature.transpose(1, 0))
        feature = feature * neg_matrix
        neg_pair = feature.masked_fill(feature < 1e-6, -np.inf)

        logits = torch.cat([pos_pair.unsqueeze(1), neg_pair], dim=1)  # (N, 1+N)
        label = torch.zeros(logits.size(0), dtype=torch.long).to(feature.device)
        loss = F.nll_loss(F.log_softmax(self.scale * logits, dim=1), label)

        if (~conf_mask).sum() != 0:
            SC_num = (class_candidates * ~conf_mask.unsqueeze(1)).sum() / (~conf_mask).sum()
        else:
            SC_num = 0

        return loss, SC_num

class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05):
        super().__init__()
        self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
        self.temp = temp

    def forward(self, x, stochastic=True):
        mu = self.mu
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu

        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        score = score / self.temp

        return score
    
    def get_proxy(self, stochastic=False):
        mu = self.mu
        
        if stochastic:
            sigma = F.softplus(self.sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu

        return weight


class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes, bias)

    def forward(self, x, stochastic=True):
        return self.linear(x)
    
    def get_proxy(self, stochastic=True):
        return self.linear.weight


@TRAINER_REGISTRY.register()
class UPCSC(TrainerXU):
    """StyleMatch for semi-supervised domain generalization.

    Reference:
        Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE

        # Inference mode: 1) deterministic 2) ensemble
        self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
        self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
        if self.inference_mode == "ensemble":
            print(f"Apply ensemble (n={self.n_ensemble}) at test time")

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.adain = AdaIN(
            cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
            cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
            self.device,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
        self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

        self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
        self.sigma_log = {"raw": [], "std": []}
        if self.save_sigma:
            assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        ############################
        self.dim = cfg.TRAINER.UPCSC.DIM
        self.upcscloss = adaptiveUPCSCLoss(cfg, self.num_classes)

        ############################

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
            self.C = StochasticClassifier(self.G.fdim, self.num_classes)
        else:
            self.C = NormalClassifier(self.G.fdim, self.num_classes, bias=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)


        print("Building Feature Projector") # fdim->dim
        self.fP = nn.Linear(self.G.fdim, self.dim)
        self.fP.to(self.device)

        
        print("# params: {:,}".format(count_num_param(self.fP)))
        self.optim_fP = build_optimizer(self.fP, cfg.TRAINER.UPCSC.P_OPTIM)
        self.sched_fP = build_lr_scheduler(self.optim_fP, cfg.TRAINER.UPCSC.P_OPTIM)
        self.register_model("fP", self.fP, self.optim_fP, self.sched_fP)


        print("Building Cls Projector") # fdim->dim
        self.cP = nn.Linear(self.G.fdim, self.dim, bias=False)
        self.cP.to(self.device)

        print("# params: {:,}".format(count_num_param(self.cP)))
        self.optim_cP = build_optimizer(self.cP, cfg.TRAINER.UPCSC.P_OPTIM)
        self.sched_cP = build_lr_scheduler(self.optim_cP, cfg.TRAINER.UPCSC.P_OPTIM)
        self.register_model("cP", self.cP, self.optim_cP, self.sched_cP)


    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                z_xu_k = self.C(self.G(xu_k), stochastic=False)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)

        ####################
        # Generate style transferred images
        ####################
        if self.apply_sty:
            xu_sty = []
            for k in range(K):
                # Content
                x_k = x0[k]
                u_k = u0[k]
                xu_k = torch.cat([x_k, u_k], 0)
                # Style
                other_domains = [i for i in range(K) if i != k]
                k2 = random.choice(other_domains)
                x_k2 = x0[k2]
                u_k2 = u0[k2]
                xu_k2 = torch.cat([x_k2, u_k2], 0)
                # Transfer
                xu_k_sty = self.adain(xu_k, xu_k2)
                xu_sty.append(xu_k_sty)

        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), stochastic=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

            # Compute loss for style transferred data
            if self.apply_sty:
                xu_k_sty = xu_sty[k]
                f_xu_k_sty = self.G(xu_k_sty)
                z_xu_k_sty = self.C(f_xu_k_sty, stochastic=True)
                loss = F.cross_entropy(z_xu_k_sty, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_sty += loss


        ########## UPCSC ##########
        xu_all = []
        xu_aug_all = []
        xu_sty_all = []
        yp_xu_all = []
        mask_all = []

        for k in range(K):
            xu_all.append(u[k])
            xu_aug_all.append(u_aug[k])
            yp_xu_all.append(p_xu.chunk(K)[k].chunk(2)[1])
            mask_all.append(mask_xu[k].chunk(2)[1])
            if self.apply_sty: xu_sty_all.append(xu_sty[k].chunk(2)[1])

            
        xu_all = torch.cat(xu_all, dim=0)
        xu_aug_all = torch.cat(xu_aug_all, dim=0)
        yp_xu_all = torch.cat(yp_xu_all, dim=0)
        mask_all = torch.cat(mask_all, dim=0)
        if self.apply_sty: xu_sty_all = torch.cat(xu_sty_all, dim=0)

        rep = []; rep_p = []; conf_mask = []
        
        # PCL Weak AUG
        rep.append(self.fP(self.G(xu_all)))
        rep_p.append(yp_xu_all)
        conf_mask.append(mask_all)

         # PCL Strong AUG
        if self.apply_aug:
            rep.append(self.fP(self.G(xu_aug_all)))
            rep_p.append(yp_xu_all)
            conf_mask.append(mask_all)
        
        if self.apply_sty:     # PCL STY
            rep.append(self.fP(self.G(xu_sty_all)))
            rep_p.append(yp_xu_all)
            conf_mask.append(mask_all)
        
        rep = torch.cat(rep, dim=0)
        rep_p = torch.cat(rep_p, dim=0)
        conf_mask = torch.cat(conf_mask, dim=0).bool()
        
        
        loss_upcsc, SC_num = self.upcscloss(rep, self.cP, self.C, rep_p, conf_mask)
                

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_upcsc
        loss_summary["loss_upcsc"] = loss_upcsc.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        if self.apply_sty:
            loss_all += loss_u_sty
            loss_summary["loss_u_sty"] = loss_u_sty.item()

        self.model_backward_and_update(loss_all)

        loss_summary["SC_num"] = SC_num
        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.save_sigma:
            sigma_raw = self.C.sigma.data  # (num_classes, num_features)
            sigma_std = F.softplus(sigma_raw - 4)
            sigma_std = sigma_std.mean(1).cpu().numpy()
            self.sigma_log["std"].append(sigma_std)
            sigma_raw = sigma_raw.mean(1).cpu().numpy()
            self.sigma_log["raw"].append(sigma_raw)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }

        return batch

    def model_inference(self, input):
        features = self.G(input)

        if self.inference_mode == "deterministic":
            prediction = self.C(features, stochastic=False)

        elif self.inference_mode == "ensemble":
            prediction = 0
            for _ in range(self.n_ensemble):
                prediction += self.C(features, stochastic=True)
            prediction = prediction / self.n_ensemble

        else:
            raise NotImplementedError

        return prediction

    def after_epoch(self):
        if (self.epoch+1) % 5 == 0 and (self.epoch+1) != self.cfg.OPTIM.MAX_EPOCH:
            self.test()
            
    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

        # Save sigma
        if self.save_sigma:
            sigma_raw = np.stack(self.sigma_log["raw"])
            np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

            sigma_std = np.stack(self.sigma_log["std"])
            np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)
