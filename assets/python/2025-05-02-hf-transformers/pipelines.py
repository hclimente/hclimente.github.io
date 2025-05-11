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
from transformers import Pipeline


# %%
class DNAPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        preprocess_params = {}

        if "max_length" in kwargs:
            preprocess_params["max_length"] = kwargs["max_length"]

        return preprocess_params, {}, {}

    def preprocess(
        self,
        model_inputs: Union[str, List[str]],
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
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
        model_inputs: Union[str, List[str]],
    ) -> Dict[str, Any]:
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
    ) -> Union[np.ndarray, List[np.ndarray]]:
        embeddings = model_outputs["hidden_states"][-1].detach()
        attention_mask = model_outputs["attention_mask"].unsqueeze(-1).cpu()
        masked_embeddings = attention_mask * embeddings

        mean_sequence_embeddings = masked_embeddings.sum(1) / attention_mask.sum(1)

        return mean_sequence_embeddings.cpu().numpy()


# %%
class DNAClassificationPipeline(DNAPipeline):

    def _forward(
        self,
        model_inputs: Union[str, List[str]],
    ) -> Dict[str, Any]:
        # find out which of the tokens are padding tokens
        # these tokens will be ignored by the model
        attention_mask = model_inputs != self.tokenizer.pad_token_id

        out = self.model(
            model_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if "attention_mask" in out:
            raise ValueError("Output contains attention_mask, " "which is unexpected.")
        out["attention_mask"] = attention_mask

        return out

    def postprocess(
        self,
        model_outputs: Dict[str, Any],
    ) -> Union[np.ndarray, List[np.ndarray]]:

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

    sequence = ["ATGGTAGCTACATCATCTG"]
    # embeddings = pipeline(sequence, max_length=33)

    # print(embeddings)

    tok_out = tokenizer(sequence, return_tensors="pt")
    print(tok_out)

    print("--- Summary with dictionary input ---")
    model(**tok_out)
    # model_summary = summary(model,
    #                         input_data={'input_ids': tok_out["input_ids"], 'attention_mask': tok_out["attention_mask"]},
    #                         col_names=["input_size", "output_size", "num_params", "mult_adds"],
    #                         verbose=1)
    # print(model_summary)
