import torch

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import add_start_docstrings, RobertaModel
from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.models.roberta.modeling_roberta import ROBERTA_START_DOCSTRING, RobertaClassificationHead, \
    ROBERTA_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC, RobertaPreTrainedModel


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassificationWithFL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config,gamma=0, alpha=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        #modifications
        self.gamma = gamma
        self.alpha = alpha

        self.init_weights()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        #processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                #modified codeiiii
                assert self.num_labels == 2, f'Expected 2 labels but found {self.num_labels}'
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                pt = torch.exp(-loss)
                if self.alpha != None:
                    alpha_tensor = torch.Tensor([self.alpha, 1 - self.alpha]).cuda()
                    at= alpha_tensor.gather(0, labels.data.view(-1))
                    loss = at * (1 - pt) ** self.gamma * loss
                else:
                    loss = (1-pt) ** self.gamma * loss
                #loss = loss.sum()i
            else:
                print('No Thing has happend. Wach out!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
'''

        if labels is not None:
            if self.config.problem_type is None:
                #if labels is not None:
                assert self.num_labels == 2, f'Expected 2 labels but found {self.num_labels}'
                
                if self.alpha != None:
                    alpha_tensor = torch.Tensor([self.alpha, 1 - self.alpha]).cuda()
                    loss_fct = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1), weight=alpha_tensor)
                else:
                    loss_fct = F.cross_entropy(logits.view(-1,self.num_labels), labels.view(-1))
                
                pt = torch.exp(-loss_fct)
                loss = (1 - pt) ** self.gamma * loss_fct
'''


