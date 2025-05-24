# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from typing import List, Dict, Tuple, Union, Optional, Any

import numpy as np
import torch as pt
from transformers import Pipeline


# %%
class DNAPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Sanitize the parameters for the pipeline.

        Args:
            **kwargs: The parameters to be sanitized.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: A tuple containing
                the sanitized parameters for preprocessing, model forward pass, and
                postprocessing, respectively.
        """
        preprocess_params = {}

        recognized_params = set(["max_length"])

        if "max_length" in kwargs:
            preprocess_params["max_length"] = kwargs["max_length"]

        unrecognized_params = set(kwargs.keys()) - recognized_params
        if unrecognized_params:
            raise ValueError(f"Unrecognized pipeline parameters: {unrecognized_params}")

        return preprocess_params, {}, {}

    def preprocess(
        self,
        model_inputs: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> List[pt.Tensor]:
        """
        Preprocess the input sequences before passing them to the model.

        Args:
            model_inputs (Union[str, List[str]]): The input sequence(s) to be tokenized.
            max_length (Optional[int]): The maximum length of the tokenized sequences.
                If None, the maximum length of the tokenizer is used.

        Returns:
            List[pt.Tensor]: The tokenized input sequences.
        """
        if max_length is None:
            max_length = self.tokenizer.model_max_length

        if isinstance(model_inputs, str):
            model_inputs = [model_inputs]

        tokens_ids = self.tokenizer.batch_encode_plus(
            model_inputs,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        return tokens_ids


# %%
class DNAEmbeddingPipeline(DNAPipeline):

    def _forward(
        self,
        model_inputs: pt.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            model_inputs (pt.Tensor): The tokenized input sequence(s).

        Returns:
            Dict[str, Any]: The model outputs.
        """
        # find out which of the tokens are padding tokens
        # these tokens will be ignored by the model
        attention_mask = model_inputs != self.tokenizer.pad_token_id

        out = self.model(
            model_inputs,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if "attention_mask" in out:
            raise ValueError("Output contains attention_mask, " "which is unexpected.")
        out["attention_mask"] = attention_mask

        return out

    def postprocess(
        self,
        model_outputs: Dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute the mean sequence embedding from the last hidden layer (size 512).

        Args:
            model_outputs (Dict[str, Any]): The model outputs.

        Returns:
            dict[str, np.ndarray]: The mean sequence embeddings for each input sequence.
        """
        out = {}

        embeddings = model_outputs["hidden_states"][-1].detach()
        attention_mask = model_outputs["attention_mask"].unsqueeze(-1).cpu()
        masked_embeddings = attention_mask * embeddings

        mean_sequence_embeddings = masked_embeddings.sum(1) / attention_mask.sum(1)
        out["embedding"] = mean_sequence_embeddings.cpu().numpy()

        return out


# %%
class DNAClassificationPipeline(DNAPipeline):

    def _forward(
        self,
        model_inputs: pt.Tensor,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            model_inputs (pt.Tensor): The tokenized input sequence(s).

        Returns:
            Dict[str, Any]: The model outputs.
        """
        # find out which of the tokens are padding tokens
        # these tokens will be ignored by the model
        attention_mask = model_inputs != self.tokenizer.pad_token_id

        out = self.model(
            model_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        out["attention_mask"] = attention_mask

        return out

    def postprocess(
        self,
        model_outputs: Dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute the logits and mean sequence embedding from the model outputs.

        Args:
            model_outputs (Dict[str, Any]): The model outputs.

        Returns:
            dict[str, np.ndarray]: The logits and mean sequence embeddings for each
                input sequence.
        """
        out = {}
        out["logits"] = model_outputs["logits"]

        embeddings = model_outputs["hidden_states"][-1].detach()
        attention_mask = model_outputs["attention_mask"].unsqueeze(-1).cpu()
        masked_embeddings = attention_mask * embeddings

        mean_sequence_embeddings = masked_embeddings.sum(1) / attention_mask.sum(1)
        out["embedding"] = mean_sequence_embeddings

        return out


# %%
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        trust_remote_code=True,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        trust_remote_code=True,
    )

    pipeline = DNAEmbeddingPipeline(model=model, tokenizer=tokenizer)

    embeddings = pipeline("ATGGTAGCTACATCATCTG")

    print(embeddings)
