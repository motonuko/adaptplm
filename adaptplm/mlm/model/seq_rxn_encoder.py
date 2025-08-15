import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import EsmModel, BertForMaskedLM, BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead, \
    BertLayer
from transformers.utils import ModelOutput

from adaptplm.mlm.model.init_util_module import InitUtilModule
from adaptplm.mlm.model.seq_rxn_encoder_config import SeqRxnEncoderConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CrossAttentionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    esm_cls_attentions: Optional[Tuple[torch.FloatTensor]] = None


# NOTE: PreTrainedModel is not used due to conceptual mismatches. we need only protein sequence encoder part after training.
# Also keeping BertModel and EsmModel separately allows to manage model and model setting easier.
class SeqRxnEncoderForMaskedLM(InitUtilModule):
    def __init__(self, esm: EsmModel, bert: BertModel, cls_head: BertOnlyMLMHead, config: SeqRxnEncoderConfig,
                 debug: bool):
        super().__init__(config)
        if debug:
            logger.setLevel(logging.DEBUG)
        self.esm = esm
        self.bert = bert
        self.config = config
        self.fnn_for_projection = nn.Linear(self.esm.config.hidden_size, self.bert.config.hidden_size)
        self.additional_bert_layers = nn.ModuleList(
            [BertLayer(self.bert.config) for _ in range(config.n_additional_layers)])
        self.cls: BertOnlyMLMHead = cls_head
        for param in self.esm.parameters():
            param.requires_grad = False
        for layer in self.esm.encoder.layer[-self.config.n_trainable_esm_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.init_weights()
        self.tie_weights()

    @classmethod
    def from_pretrained_models(cls, esm_pretrained, bert_pretrained, config: SeqRxnEncoderConfig,
                               resized_token_embedding_size: Optional[int] = None, debug: bool = False):
        esm = EsmModel.from_pretrained(esm_pretrained)
        bert_for_mlm = BertForMaskedLM.from_pretrained(bert_pretrained)
        print("Pretrained BERT model has loaded")
        if resized_token_embedding_size:
            bert_for_mlm.resize_token_embeddings(resized_token_embedding_size)
        bert_model = bert_for_mlm.bert
        cls_head = bert_for_mlm.cls
        return SeqRxnEncoderForMaskedLM(esm, bert_model, cls_head, config, debug)

    @classmethod
    def from_pretrained_esm(cls, esm_pretrained, bert_config: Path, config: SeqRxnEncoderConfig, debug: bool = False):
        esm = EsmModel.from_pretrained(esm_pretrained)
        bert_config = BertConfig.from_json_file(bert_config)
        bert_for_mlm = BertForMaskedLM(bert_config)
        logging.info("BERT model will be trained from scratch.")
        bert_model = bert_for_mlm.bert
        cls_head = bert_for_mlm.cls
        return SeqRxnEncoderForMaskedLM(esm, bert_model, cls_head, config, debug)

    @classmethod
    def from_configs(cls, esm_config, bert_config, model_config: SeqRxnEncoderConfig):
        esm = EsmModel(esm_config)
        bert_model = BertModel(bert_config)
        cls_head = BertOnlyMLMHead(bert_config)
        return SeqRxnEncoderForMaskedLM(esm, bert_model, cls_head, model_config)

    def get_input_embeddings(self):
        # ref: BertModel
        return self.bert.embeddings.word_embeddings

    def get_output_embeddings(self):
        # ref: BertForMaskedLM
        return self.cls.predictions.decoder

    def get_parameters(self):
        trained_params = (
            # NOTE: ESM encoder layers except last layer are set as `require_grad=False`
                list([param for layer in self.esm.encoder.layer[:- self.config.n_trainable_esm_layers] for param in
                      layer.parameters()])
                + list(self.esm.contact_head.parameters())
                + list(self.esm.embeddings.parameters())
                + list(self.bert.encoder.parameters())
                + list(self.bert.embeddings.position_embeddings.parameters())
                + list(self.bert.embeddings.token_type_embeddings.parameters())
                + list(self.bert.embeddings.LayerNorm.parameters())
        )
        esm_upper_layer_params = (list(
            [param for layer in self.esm.encoder.layer[-self.config.n_trainable_esm_layers:] for param in
             layer.parameters()])
                                  # NOTE: emb_layer_norm_after is used after all encoder layers.
                                  + list(self.esm.encoder.emb_layer_norm_after.parameters())
                                  )
        untrained_params = (  # new prams or
                list(self.esm.pooler.parameters())
                + list(self.bert.embeddings.word_embeddings.parameters())
                + list(self.fnn_for_projection.parameters())
                + list(self.additional_bert_layers.parameters())
                + [self.cls.predictions.bias]  # cls.predictions.decoder is tied with bert.embeddings.word_embeddings
                + list(self.cls.predictions.transform.parameters())  # questionable to retrain
        )
        # If you specify the parameters twice and pass them to the optimizer, for some reason you get different results.
        assert len(set(trained_params)) == len(list(trained_params))
        assert len(set(untrained_params)) == len(list(untrained_params))
        assert len(set(esm_upper_layer_params)) == len(list(esm_upper_layer_params))
        assert len(set(trained_params).intersection(set(untrained_params))) == 0
        assert len(set(trained_params).intersection(set(esm_upper_layer_params))) == 0
        assert len(set(untrained_params).intersection(set(esm_upper_layer_params))) == 0
        all_params_merged = set(list(trained_params) + list(untrained_params) + list(esm_upper_layer_params))
        all_params = set(self.parameters())
        assert all_params_merged == all_params, f"{len(all_params_merged)}, {len(all_params)}"
        return trained_params, untrained_params, esm_upper_layer_params

    def get_parameters_for_untrained_bert(self):
        trained_params = (
            # NOTE: ESM encoder layers except last layer are set as `require_grad=False`
                list([param for layer in self.esm.encoder.layer[:-self.config.n_trainable_esm_layers] for param in
                      layer.parameters()])
                + list(self.esm.contact_head.parameters())
                + list(self.esm.embeddings.parameters())
        )
        esm_upper_layer_params = (list(
            [param for layer in self.esm.encoder.layer[-self.config.n_trainable_esm_layers:] for param in
             layer.parameters()])
                                  # NOTE: emb_layer_norm_after is used after all encoder layers.
                                  + list(self.esm.encoder.emb_layer_norm_after.parameters())
                                  )
        untrained_params = (  # new prams or
                list(self.esm.pooler.parameters())
                + list(self.bert.embeddings.word_embeddings.parameters())
                + list(self.bert.encoder.parameters())
                + list(self.bert.embeddings.position_embeddings.parameters())
                + list(self.bert.embeddings.token_type_embeddings.parameters())
                + list(self.bert.embeddings.LayerNorm.parameters())
                + list(self.fnn_for_projection.parameters())
                + list(self.additional_bert_layers.parameters())
                + [self.cls.predictions.bias]  # cls.predictions.decoder is tied with bert.embeddings.word_embeddings
                + list(self.cls.predictions.transform.parameters())  # questionable to retrain
        )
        # If you specify the parameters twice and pass them to the optimizer, for some reason you get different results.
        assert len(set(trained_params)) == len(list(trained_params))
        assert len(set(untrained_params)) == len(list(untrained_params))
        assert len(set(esm_upper_layer_params)) == len(list(esm_upper_layer_params))
        assert len(set(trained_params).intersection(set(untrained_params))) == 0
        assert len(set(trained_params).intersection(set(esm_upper_layer_params))) == 0
        assert len(set(untrained_params).intersection(set(esm_upper_layer_params))) == 0
        all_params_merged = set(list(trained_params) + list(untrained_params) + list(esm_upper_layer_params))
        all_params = set(self.parameters())
        assert all_params_merged == all_params, f"{len(all_params_merged)}, {len(all_params)}"
        return trained_params, untrained_params, esm_upper_layer_params

    def forward(
            self,
            aaseq_input_ids: Optional[torch.Tensor] = None,
            aaseq_attention_mask: Optional[torch.Tensor] = None,
            aaseq_labels: Optional[torch.Tensor] = None,
            # protein sequence is not masked so this won't be used.
            smiles_input_ids: Optional[torch.Tensor] = None,
            smiles_attention_mask: Optional[torch.Tensor] = None,
            smiles_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], Any]:

        logger.debug(
            f"aaseq_input_ids shape: {aaseq_input_ids.shape}, smiles_input_ids shape: {smiles_input_ids.shape}")
        esm_output = self.esm(input_ids=aaseq_input_ids, attention_mask=aaseq_attention_mask, output_attentions=True)

        esm_attentions = esm_output.attentions
        esm_cls_attentions = tuple(layer[:, :, 0, :] for layer in esm_attentions)

        aa_seq_feature = esm_output.pooler_output
        aa_seq_feature = torch.unsqueeze(aa_seq_feature, dim=1)

        bert_output = self.bert(
            input_ids=smiles_input_ids,
            attention_mask=smiles_attention_mask)

        aa_seq_feature = self.fnn_for_projection(aa_seq_feature)

        input_shape = smiles_input_ids.size()
        extended_smiles_attn_mask = self.bert.get_extended_attention_mask(smiles_attention_mask, input_shape)

        for additional_layer in self.additional_bert_layers:
            hidden_states = bert_output.last_hidden_state + aa_seq_feature
            layer_outputs = additional_layer(
                hidden_states=hidden_states,  # [batch, token, feat]
                attention_mask=extended_smiles_attn_mask,
            )
            hidden_states = layer_outputs[0]
        # cross_attention_output = self.additional_bert_layer(
        #     hidden_states=updated_hidden,  # [batch, token, feat]
        #     attention_mask=extended_smiles_attn_mask,
        # )
        # hidden_states = cross_attention_output[0]
        smiles_prediction_scores = self.cls(hidden_states)
        masked_lm_loss = None
        if smiles_labels is not None:
            # -100 (should be unmasked tokens) will be ignored by default.
            # Check out the implementation of DataCollatorForLanguageModeling.torch_mask_tokens
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(smiles_prediction_scores.view(-1, self.bert.config.vocab_size),
                                      smiles_labels.view(-1))

        return CrossAttentionModelOutput(loss=masked_lm_loss,
                                         logits=smiles_prediction_scores,
                                         esm_cls_attentions=esm_cls_attentions,
                                         )
