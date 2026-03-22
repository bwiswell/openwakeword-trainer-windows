# Copyright 2022 David Scripka. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################

# NOTICE: This file has been substantially modified by Benj Wiswell.


from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch as to
import torchmetrics as tm
import tqdm as tq

from ..config import Config
from ..data_manager import DataManager
from ..logger import Logger

from .mmap_dataset import MMapDataset
from .network import Network


DataSplit = to.utils.data.DataLoader[tuple[to.FloatTensor, to.FloatTensor]]


class Trainer(to.nn.Module):

    VAL_FP_HRS = 11.3

    def __init__ (self, config: Config, dm: DataManager):
        self.config = config
        self.dm = dm

        self.device = to.device('cuda:0' if to.cuda.is_available() else 'cpu')
        self.input_shape = self._get_input_shape()
        self.layer_dim = config.layer_size
        self.secs_per_example = 1280 * self.input_shape[0] / 16000

        self.best_models: list[Network] = []
        self.best_model_scores: list[float] = []
        self.best_val_fp = 1000.0
        self.best_val_accuracy = 0.0
        self.best_val_recall = 0.0
        self.best_train_recall = 0.0

        self.model = Network(self.input_shape, self.layer_dim)

        self.fp = lambda pred, y: (y - pred <= -0.5).sum()
        self.accuracy = tm.Accuracy(task = 'binary')
        self.recall = tm.Recall(task = 'binary')

        self.n_fp = 0
        self.val_fp = 0

        self.history = defaultdict(list)

        self.loss = to.nn.functional.binary_cross_entropy
        self.optimizer = to.optim.Adam(self.model.parameters(), lr = 0.0001)


    ### HELPERS ###
    def _average_models (
                self,
                models: Optional[list[Network]] = None
            ) -> Network:
        if models is None:
            models = self.best_models

        # Clone a model from the list as the base for the averaged model
        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        # Initialize a running total of the weights
        for key in averaged_model_dict:
            averaged_model_dict[key] *= 0  # set to 0

        for model in models:
            model_dict = model.state_dict()
            for key, value in model_dict.items():
                averaged_model_dict[key] += value

        for key in averaged_model_dict:
            averaged_model_dict[key] /= len(models)

        # Load the averaged weights into the model
        averaged_model.load_state_dict(averaged_model_dict)

        return averaged_model


    def _get_input_shape (self) -> tuple[int]:
        return np.load(self.dm.features.pos_test).shape[1:]


    def _load_data_splits (self) -> tuple[DataSplit, DataSplit, DataSplit]:
        x_train = to.utils.data.DataLoader(
            MMapDataset(self.config, self.dm, self.input_shape[0]),
            batch_size = None,
            num_workers = 0,
            prefetch_factor = None
        )

        x_val_fp_x: npt.NDArray[np.float32] = np.load(
            self.dm.features.validation
        )
        x_val_fp = to.utils.data.DataLoader(
            to.utils.data.TensorDataset(
                to.from_numpy(np.array([
                    [x_val_fp_x[i:i + self.input_shape[0]]]
                    for i in range(x_val_fp_x.shape[0] - self.input_shape[0])
                ])),
                to.from_numpy(np.zeros(x_val_fp_x.shape[0], np.float32)),
            ),
            batch_size = 100
        )

        x_val_pos_x: npt.NDArray[np.float32] = np.load(
            self.dm.features.pos_test
        )
        x_val_neg_x: npt.NDArray[np.float32] = np.load(
            self.dm.features.neg_test
        )
        x_val = to.utils.data.DataLoader(
            to.utils.data.TensorDataset(
                to.from_numpy(np.vstack((x_val_pos_x, x_val_neg_x))),
                to.from_numpy(np.hstack((
                    np.ones(x_val_pos_x.shape[0], np.float32),
                    np.zeros(x_val_neg_x.shape[0], np.float32)
                )))
            ),
            batch_size = 100
        )

        return x_train, x_val_fp, x_val



    def _lr_warmup_cos_decay (
                self,
                global_step: int,
                warmup_steps: int = 0,
                hold: int = 0,
                total_steps: int = 0,
                target_lr: float = 1e-3
            ) -> float:
        num = global_step - warmup_steps - hold
        denom = float(total_steps - warmup_steps - hold)

        lr = 0.5 * target_lr * (1 + np.cos(np.pi * num / denom))
        warmup_lr = target_lr * (global_step / warmup_steps)

        if hold > 0:
            lr = np.where(global_step > warmup_steps + hold, lr, target_lr)

        lr = np.where(global_step < warmup_steps, warmup_lr, lr)
        return lr
    

    def _train (
                self,
                x: DataSplit,
                x_val: DataSplit,
                x_val_fp: DataSplit,
                neg_weight_sched: list[float],
                val_steps: npt.NDArray[np.int64],
                lr: float,
                desc: str = 'Training model'
            ):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        # Train model
        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = to.Tensor([]).to(self.device)
        accumulated_labels = to.Tensor([]).to(self.device)

        pbar = tq.tqdm(enumerate(x), total=self.config.steps, desc=desc)
        for step_idx, (x, y) in pbar:
            # Move the inputs to the target device
            x: to.FloatTensor = x.to(self.device)
            y: to.FloatTensor = y.to(self.device)

            # Update learning rates
            for g in self.optimizer.param_groups:
                g['lr'] = self._lr_warmup_cos_decay(
                    step_idx,
                    warmup_steps = self.config.steps // 5,
                    hold_steps = self.config.steps // 3,
                    total_steps = self.config.steps,
                    target_lr = lr
                )

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Get predictions for batch
            preds: to.FloatTensor = self.model(x)

            # Construct batch with only samples that have high loss
            preds = to.cat((
                preds[(y == 0) & (preds.squeeze() >= 0.001)],
                preds[(y == 1) & (preds.squeeze() < 0.999)]
            ))
            y = to.cat((
                y[(y == 0) & (preds.squeeze() >= 0.001)],
                y[(y == 1) & (preds.squeeze() < 0.999)]
            ))
            y_p = y[..., None]

            # Set weights for batch
            if len(neg_weight_sched) == 1:
                w = to.ones(y.shape[0]) * neg_weight_sched[0]
                pos_idxs = y == 1
                w[pos_idxs] = 1
                w = w[..., None]
            else:
                w = to.ones(y.shape[0]) * neg_weight_sched[step_idx]
                pos_idxs = y == 1
                w[pos_idxs] = 1
                w = w[..., None]

            # Do backpropagation, with gradient accumulation if the batch-size
            # after selecting high loss examples is too small
            if preds.shape[0] != 0:
                loss = self.loss(preds, y_p, w.to(self.device))
                loss = loss / accumulation_steps
                accumulated_samples += preds.shape[0]

                if preds.shape[0] >= 128:
                    accumulated_predictions = preds
                    accumulated_labels = y_p
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = to.cat(
                        (accumulated_predictions, preds)
                    )
                    accumulated_labels = to.cat((accumulated_labels, y_p))
                else:
                    loss.backward()
                    self.optimizer.step()
                    accumulation_steps = 1
                    accumulated_samples = 0

                    self.history["loss"].append(loss.detach().cpu().numpy())

                    # Compute training metrics and log them
                    fp = self.fp(accumulated_predictions, accumulated_labels)
                    self.n_fp += fp
                    self.history["recall"].append(self.recall(
                        accumulated_predictions, accumulated_labels
                    ).detach().cpu().numpy())

                    accumulated_predictions = to.Tensor([]).to(self.device)
                    accumulated_labels = to.Tensor([]).to(self.device)

            # Run validation and log validation metrics
            if step_idx in val_steps and step_idx > 1 and x_val_fp is not None:
                # Get false positives per hour with false positive data
                val_fp = 0
                for data in x_val_fp:
                    with to.no_grad():
                        x_val: to.FloatTensor = data[0].to(self.device)
                        y_val: to.FloatTensor = data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_fp += self.fp(val_predictions, y_val[..., None])
                val_fp_per_hr = (
                    val_fp / Trainer.VAL_FP_HRS
                ).detach().cpu().numpy()
                self.history["val_fp_per_hr"].append(val_fp_per_hr)

            if step_idx in val_steps and step_idx > 1 and x_val is not None:
                # Get metrics for balanced test examples of positive and
                # negative clips
                for data in x_val:
                    with to.no_grad():
                        x_val: to.FloatTensor = data[0].to(self.device)
                        y_val: to.FloatTensor = data[1].to(self.device)
                        val_predictions = self.model(x_val)
                        val_recall = self.recall(
                            val_predictions, y_val[..., None]
                        ).detach().cpu().numpy()
                        val_acc = self.accuracy(
                            val_predictions, y_val[..., None].to(to.int64)
                        )
                        val_fp = self.fp(val_predictions, y_val[..., None])
                self.history["val_accuracy"].append(
                    val_acc.detach().cpu().numpy()
                )
                self.history["val_recall"].append(val_recall)
                self.history["val_n_fp"].append(
                    val_fp.detach().cpu().numpy()
                )

            # Save models with a validation score above/below the 90th percentile
            # of the validation scores up to that point
            if step_idx in val_steps and step_idx > 1:
                val_n_fp_perc = np.percentile(self.history["val_n_fp"], 50)
                val_recall_perc = np.percentile(self.history["val_recall"], 5)
                if self.history["val_n_fp"][-1] <= val_n_fp_perc and \
                        self.history["val_recall"][-1] >= val_recall_perc:
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({
                        "training_step_ndx": step_idx,
                        "val_n_fp": self.history["val_n_fp"][-1],
                        "val_recall": self.history["val_recall"][-1],
                        "val_accuracy": self.history["val_accuracy"][-1],
                        "val_fp_per_hr": self.history.get(
                            "val_fp_per_hr", [0]
                        )[-1]
                    })
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]

            if step_idx == self.config.steps - 1:
                break
    

    ### METHODS ###
    def auto_train (self) -> Network:
        '''
        A sequence of training steps that produce relatively strong models
        automatically, based on validation data and performance targets
        provided. After training merges the best checkpoints and returns a
        single model.
        '''

        x_train, x_val_fp, x_val = self._load_data_splits()
        max_neg_weight = self.config.max_neg_weight

        # Sequence 1
        Logger.log('Starting training sequence 1...')
        lr = 0.0001
        weights = np.linspace(1, max_neg_weight, int(steps)).tolist()
        val_steps = np.linspace(
            steps - int(steps*0.25), steps, 20
        ).astype(np.int64)
        self._train(
            x = x_train,
            x_val = x_val,
            x_val_fp = x_val_fp,
            neg_weight_sched = weights,
            val_steps = val_steps,
            lr = lr,
            desc = 'Training sequence 1'
        )

        # Sequence 2
        Logger.log('Starting training sequence 2...')
        lr /= 10
        steps /= 10

        # Adjust weights as needed based on false positive per hour performance
        # from first sequence
        if self.best_val_fp > self.config.target_fp:
            max_neg_weight *= 2

        weights = np.linspace(1, max_neg_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self._train(
            x = x_train,
            x_val = x_val,
            x_val_fp = x_val_fp,
            neg_weight_sched = weights,
            val_steps = val_steps,
            lr = lr,
            desc = 'Training sequence 2'
        )

        # Sequence 3
        Logger.log('Starting training sequence 3...')
        lr /= 10

        # Adjust weights as needed based on false positive per hour performance
        # from second sequence
        if self.best_val_fp > self.config.target_fp:
            max_neg_weight *= 2

        weights = np.linspace(1, max_neg_weight, int(steps)).tolist()
        val_steps = np.linspace(1, steps, 20).astype(np.int16)
        self._train(
            x = x_train,
            x_val = x_val,
            x_val_fp = x_val_fp,
            neg_weight_sched = weights,
            val_steps = val_steps,
            lr = lr,
            desc = 'Training sequence 3'
        )

        # Merge best models
        Logger.log('Merging 90th percentile checkpoints into a single model')
        accuracy_percentile = np.percentile(self.history["val_accuracy"], 90)
        recall_percentile = np.percentile(self.history["val_recall"], 90)
        fp_percentile = np.percentile(self.history["val_fp_per_hr"], 10)

        # Get models above the 90th percentile
        models: list[Network] = []
        for model, score in zip(self.best_models, self.best_model_scores):
            if score["val_accuracy"] >= accuracy_percentile and \
                    score["val_recall"] >= recall_percentile and \
                    score["val_fp_per_hr"] <= fp_percentile:
                models.append(model)

        if len(models) > 0:
            combined_model = self._average_models(models)
        else:
            combined_model = self.model

        # Report validation metrics for combined model
        with to.no_grad():
            for batch in x_val:
                x: to.FloatTensor = batch[0].to(self.device)
                y: to.FloatTensor = batch[1].to(self.device)
                val_ps = combined_model(x)

            combined_model_recall = self.recall(
                val_ps, y[..., None]
            ).detach().cpu().numpy()
            combined_model_accuracy = self.accuracy(
                val_ps, y[..., None].to(to.int64)
            ).detach().cpu().numpy()

            combined_model_fp = 0
            for batch in x_val_fp:
                x_val: to.FloatTensor = batch[0].to(self.device)
                y_val: to.FloatTensor = batch[1].to(self.device)
                val_ps = combined_model(x_val)
                combined_model_fp += self.fp(val_ps, y_val[..., None])

            combined_model_fp_per_hr = (
                combined_model_fp / Trainer.VAL_FP_HRS
            ).detach().cpu().numpy()

        Logger.log(f'Final accuracy: {combined_model_accuracy}')
        Logger.log(f'Final recall: {combined_model_recall}')
        Logger.log(f'Final FP per hour: {combined_model_fp_per_hr}')

        stats = {
            'accuracy': combined_model_accuracy.tolist(),
            'fpah': combined_model_fp_per_hr.tolist(),
            'recall': combined_model_recall.tolist()
        }

        with open(self.dm.stats, 'w') as f:
            json.dump(stats, f)

        return combined_model



    def export (self, model: Network, name: str, output_dir: Path):
        out_model = copy.deepcopy(model)
        to.onnx.export(
            out_model.to('cpu'),
            to.rand(self.input_shape)[None, ],
            str(output_dir / f'{name}.onnx'),
            opset_version = 13
        )