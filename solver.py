import ast
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.cuda.amp as amp

from model.CPMAE import CPMAE
from data_factory.data_loader import get_loader_segment
from evaluation.evaluator import Evaluator


# ==============================================================================
# Early Stopping Utility
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, dataset_name=''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, optimizer, epoch, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, epoch, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch, path):
        if self.verbose:
            print(f'Validation score decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(path, f"{self.dataset}_checkpoint.pth"))
        self.val_loss_min = val_loss


# ==============================================================================
# Main Solver
# ==============================================================================
class Solver(object):
    DEFAULTS = {
        'patience': 5  # Default patience if not provided in config
    }

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)

        self.num_patch = self._parse_scale_arg(self.num_patch)
        self.num_patches_tf = self._parse_scale_arg(self.num_patches_tf)

        if isinstance(self.num_patches_tf, int) and self.num_patches_tf <= 0:
            self.num_patches_tf = None

        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            train_split=self.train_split,
            mode='train',
            data_name=self.dataset
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            train_split=self.train_split,
            mode='val',
            data_name=self.dataset
        )
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            train_split=self.train_split,
            mode='test',
            data_name=self.dataset
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            train_split=self.train_split,
            mode='thre',
            data_name=self.dataset
        )

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scaler = amp.GradScaler(enabled=True)  # Mixed precision scaler

        self.build_model()

    def _parse_scale_arg(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
            if "," in value:
                return [int(v.strip()) for v in value.split(",") if v.strip() != ""]
            if value.startswith("[") or value.startswith("("):
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return [int(v) for v in parsed]
                return int(parsed)
            return int(value)
        return value

    def build_model(self):
        # Accelerate fixed-size input operations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.model = CPMAE(
            win_size=self.win_size,
            n_features=self.input_c,
            num_patches=self.num_patch,
            num_patches_tf=self.num_patches_tf,
            d_model=self.d_model,
            e_layers=self.e_layers,
            alpha=self.alpha,
            beta=self.beta,
            dev=self.device,
            st_mask_ratio=self.st_mask_ratio,
            tf_mask_ratio=self.tf_mask_ratio,
            mc_samples=self.mc_samples,
            mc_mask_ratio_time=self.mc_mask_ratio_time,
            mc_mask_ratio_freq=self.mc_mask_ratio_freq,
            uncertainty_weight=self.gamma
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _eval_forward_kwargs(self, mc_samples):
        return dict(
            mc_samples=mc_samples,
            mc_mask_ratio_time=self.mc_mask_ratio_time,
            mc_mask_ratio_freq=self.mc_mask_ratio_freq,
            uncertainty_weight=self.gamma
        )

    def _student_ckpt_path(self):
        return os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')

    def _load_checkpoint(self, resume_training=False, strict=True):
        ckpt_path = self._student_ckpt_path()
        if not os.path.exists(ckpt_path):
            if strict:
                raise FileNotFoundError(f"Error: Checkpoint not found at {ckpt_path}.")
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}. Starting from scratch.")
                return 0

        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if 'model_state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
            return 0

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if resume_training and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0)

    def vali(self, vali_loader):
        self.model.eval()
        score_list = []

        with torch.no_grad():
            for input_data in vali_loader:
                input_data = input_data.float().to(self.device, non_blocking=True)

                res_dic = self.model(
                    input_data,
                    **self._eval_forward_kwargs(mc_samples=max(self.mc_samples // 4, 2))
                )

                val_score = res_dic['score'].mean().item()
                score_list.append(val_score)

        return np.average(score_list)

    def train(self, resume=False):
        print("======================TRAIN MODE======================")
        os.makedirs(self.model_save_path, exist_ok=True)

        start_epoch = 0
        if resume:
            start_epoch = self._load_checkpoint(resume_training=True, strict=False)
            print(f"Resuming training from epoch {start_epoch + 1}")

        early_stopping = EarlyStopping(
            patience=getattr(self, 'patience', 5),
            verbose=True,
            dataset_name=self.dataset
        )

        train_steps = len(self.train_loader)
        start_time = time.perf_counter()

        for epoch in tqdm(range(start_epoch, self.num_epochs)):
            loss_list = []
            self.model.train()
            epoch_time = time.time()
            with tqdm(total=train_steps) as pbar:
                for input_data in self.train_loader:
                    self.optimizer.zero_grad(set_to_none=True)
                    input_data = input_data.float().to(self.device, non_blocking=True)

                    with amp.autocast(enabled=True):
                        loss_dict = self.model(input_data)
                        loss = loss_dict['loss']

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    loss_list.append(loss.item())
                    pbar.update(1)
            train_cost_time = time.time() - epoch_time
            print("Epoch: {0}, Train Cost time: {1:.3f}s ".format(epoch + 1, train_cost_time))
            train_loss = np.average(loss_list)
            vali_score = self.vali(self.vali_loader)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | Vali Score: {vali_score:.7f}")

            # Check early stopping and save best model
            early_stopping(vali_score, self.model, self.optimizer, epoch + 1, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping triggered. Terminating training.")
                break

        end_time = time.perf_counter()
        print(f"Total training time: {end_time - start_time:.2f} seconds")

    def test(self):
        self._load_checkpoint(resume_training=False, strict=True)
        self.model.eval()

        print("======================TEST MODE======================")

        attens_energy = []
        with torch.no_grad():
            for input_data in self.thre_loader:
                input_data = input_data.float().to(self.device, non_blocking=True)
                with amp.autocast(enabled=False):
                    res_dic = self.model(input_data, **self._eval_forward_kwargs(mc_samples=self.mc_samples))
                cri = res_dic['score'].detach().cpu().numpy()
                attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        thresh = np.percentile(test_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        input_data_list = []

        with torch.no_grad():
            for input_data, labels in self.test_loader:
                input_data = input_data.float().to(self.device, non_blocking=True)

                with amp.autocast(enabled=False):
                    res_dic = self.model(input_data, **self._eval_forward_kwargs(mc_samples=self.mc_samples))

                cri = res_dic['score'].detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels.numpy())
                input_data_list.append(input_data.detach().cpu().numpy())

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred_raw = (test_energy > thresh).astype(int)

        metrics_score = ["R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR", "auc_roc"]
        evaluator_score = Evaluator(metrics_score)
        results_score = evaluator_score.evaluate(test_labels, test_energy)

        metrics_label = [
            "accuracy", "precision", "recall", "f_score",
            "affiliation_precision", "affiliation_recall", "affiliation_f",
            "adjust_precision", "adjust_recall", "adjust_f_score"
        ]
        evaluator_label = Evaluator(metrics_label)
        results_label = evaluator_label.evaluate(test_labels, pred_raw)

        metrics_label.extend(metrics_score)
        results_label.extend(results_score)

        results_np = np.array(results_label).reshape(1, -1).round(4)
        results_pd = pd.DataFrame(data=results_np, columns=metrics_label)

        for key, value in results_pd.items():
            print('{} : {}'.format(key, value.values[0]))

        return results_pd

    def plot_visualization(self, idx, input_data, res_dic, labels, batch_idx, save_dir="visualizations"):
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        x = input_data[idx].detach().cpu().numpy()
        err_time = res_dic['err_time_recon'][idx].detach().cpu().numpy()
        err_freq = res_dic['err_freq_recon'][idx].detach().cpu().numpy()
        err_unc = res_dic['err_uncertainty'][idx].detach().cpu().numpy()
        score = res_dic['score'][idx].detach().cpu().numpy()
        label = labels[idx].detach().cpu().numpy().flatten()

        T, C = x.shape
        channel_std = np.std(x, axis=0)
        c_vis = int(np.argmax(channel_std))

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axes[0].set_title(f"Input Signal (Channel {c_vis})", fontsize=12)
        axes[0].plot(x[:, c_vis], label="Input", color="black", linewidth=1.5)
        axes[0].legend(loc='upper right')

        axes[1].set_title("Branch-wise Reconstruction Error", fontsize=12)
        axes[1].plot(err_time, label="Time Error", color="blue", alpha=0.8)
        axes[1].plot(err_freq, label="Frequency Error", color="orange", alpha=0.8)
        axes[1].legend(loc='upper right')

        axes[2].set_title("Uncertainty Error", fontsize=12)
        axes[2].plot(err_unc, label="Uncertainty", color="green")
        axes[2].legend(loc='upper right')

        axes[3].set_title("Final Anomaly Score vs Ground Truth", fontsize=12)
        axes[3].plot(score, label="Score", color="red", linewidth=1.5)
        axes[3].fill_between(
            range(T),
            0,
            score.max() * 1.1 + 1e-5,
            where=(label == 1),
            color='red',
            alpha=0.2,
            label="True Anomaly"
        )
        axes[3].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"batch_{batch_idx}_{idx}_{self.dataset}_visual.png"),
            dpi=150
        )
        plt.close()