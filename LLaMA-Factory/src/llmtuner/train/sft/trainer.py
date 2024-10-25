import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
import tempfile
import shutil


if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp_dir = tempfile.mkdtemp(dir='./tmp/')  # Create a temporary directory
        self.counter = 0

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        forward_output = model(**inputs, output_hidden_states=True)

        self.save_hidden_states_to_disk(forward_output.hidden_states)

        # forward_output.hidden_states = [num_layer, batch_size, prompt_len, hidden_dim]
       
        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))
    
    def save_hidden_states(self, task, layer):
        array_all = []
        for i in range(self.counter):
            layer_file = os.path.join(self.tmp_dir, f"hidden_states_layer_{layer}_{i}.npy")
            with open(layer_file, 'rb') as f:
                hidden_states_all = np.load(f) # this is one data point's representation in one layer
                array_all.append(torch.from_numpy(hidden_states_all))

        hidden_states = []
        for hidden_states_example in array_all:
            
            # only save last token in the prompt
            hidden_tmp = hidden_states_example[:,-1,:] # [batch_size, 1 (last token from prompt), hidden_dim]
            hidden_tmp = hidden_tmp.view(-1, hidden_tmp.shape[-1])# [batch_size, hidden_dim]

            # # save all tokens in the prompt
            # hidden_tmp = hidden_states_example # [batch_size, seq_len, hidden_dim]
            # hidden_tmp = hidden_tmp.view(-1, hidden_tmp.shape[-1])# [batch_size * seq_len, hidden_dim]

            hidden_states.append(hidden_tmp)
        hidden = torch.cat(hidden_states, dim=0) # [total_data_size, hidden_size]
        hidden_state_file = os.path.join(self.args.output_dir, task, "hidden_states_{}.npy".format(layer))
        logger.info(f"Saving hidden states to {hidden_state_file}")
        if not os.path.exists(os.path.join(self.args.output_dir, task)):
            os.makedirs(os.path.join(self.args.output_dir, task))
        np.save(hidden_state_file, hidden.numpy())
    
    def save_hidden_states_to_disk(self, hidden_states):
        for layer_idx, hs_layer in enumerate(hidden_states):
            layer_file = os.path.join(self.tmp_dir, f"hidden_states_layer_{layer_idx}_{self.counter}.npy")
            hs_cpu = hs_layer.cpu() 

            # only save last token in the prompt
            hs_cpu = hs_cpu[:,-1,:] # [batch_size, 1 (last token from prompt), hidden_dim]
            hs_cpu = hs_cpu.unsqueeze(1) # insert a dim in position 1

            # # only save average of all task token in the prompt
            # hs_cpu = hs_cpu[:,133:,:] # [batch_size, prompt_len (all task tokens excluding system prompt which has 133 tokens), hidden_dim]
            # hs_cpu = torch.mean(hs_cpu, dim=1, keepdim=True)

            # # only save task tokens in the prompt
            # # llama2 system prompt has 133 tokens 
            # hs_cpu = hs_cpu[:,133:,:] # [batch_size, prompt_len (all task tokens excluding system prompt), hidden_dim]

            np.save(layer_file, hs_cpu.numpy())
        self.counter += 1

