from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
import copy
from typing import List, Dict, Optional, Union, Tuple
import os


class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.model_name_or_path = model_name_or_path
        if model_name_or_path=='bi-contriever':
            model_name_or_path = "facebook/contriever"
        if model_name_or_path.startswith('bigtr'):
            model_name_or_path = model_name_or_path.split('#')[1]
        if 'bigtr' in model_name_or_path and os.path.isdir(model_name_or_path):
            config = AutoConfig.from_pretrained(os.path.join(model_name_or_path,'with_prompt'), **model_args, cache_dir=cache_dir)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model(self.model_name_or_path, config, cache_dir, **model_args)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        #No max_seq_length set. Try to infer from model
        # print('max_seq_length ', max_seq_length)
        max_seq_length = 512
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        print('max_seq_length ',max_seq_length)

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__


    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        self.model_type = model_name_or_path
        if model_name_or_path=='bi-contriever':
            print('bi-contriever model')
            from src.contriever import Contriever
            self.query_model = Contriever.from_pretrained("facebook/contriever",cache_dir=cache_dir)
            self.doc_model = Contriever.from_pretrained("facebook/contriever",cache_dir=cache_dir)
        elif 'contriever' in model_name_or_path and 'facebook' in model_name_or_path:
            print('contriever model')
            from src.contriever import Contriever
            self.auto_model = Contriever.from_pretrained("facebook/contriever",cache_dir=cache_dir)
        elif model_name_or_path.startswith('bigtr') and not os.path.isdir(model_name_or_path):
            print('use bigtr without training')
            real_model_name_or_path = model_name_or_path.split('#')[1]
            from transformers import T5EncoderModel
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            self.model_with_prompt = T5EncoderModel.from_pretrained(real_model_name_or_path, config=config, cache_dir=cache_dir,
                                                             **model_args)
            self.model_without_prompt = copy.deepcopy(self.model_with_prompt)
        elif 'bigtr' in model_name_or_path and os.path.isdir(model_name_or_path):
            print('use bigtr with training')
            from transformers import T5EncoderModel
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            self.model_with_prompt = T5EncoderModel.from_pretrained(os.path.join(model_name_or_path,'with_prompt'), config=config,
                                                                    cache_dir=cache_dir,
                                                                    **model_args)
            self.model_without_prompt = T5EncoderModel.from_pretrained(os.path.join(model_name_or_path,'without_prompt'), config=config,
                                                                    cache_dir=cache_dir,
                                                                    **model_args)
        elif isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _load_t5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def _load_mt5_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from T5"""
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)

    def __repr__(self):
        return "Transformer"

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        # print(features)
        # exit(0)
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        # print('self.model_type', self.model_type)
        if self.model_type=='bi-contriever':
            if features['text_type']==0:
                # print('encode query')
                if 'context_masks' in features:
                    trans_features['context_masks'] = features['context_masks']
                emb = self.query_model(**trans_features)
                return {'sentence_embedding': emb}
            elif features['text_type']==1:
                # print('encode doc')
                assert not 'context_masks' in features
                emb = self.doc_model(**trans_features)
                return {'sentence_embedding': emb}
        elif 'contriever' in self.model_type and 'facebook' in self.model_type:
            if 'context_masks' in features:
                trans_features['context_masks'] = features['context_masks']
            emb = self.auto_model(**trans_features)
            return {'sentence_embedding': emb}
        elif self.model_type.startswith('bigtr') or ('bigtr' in self.model_type and os.path.isdir(self.model_type)):
            context_masks = None
            assert 'context_masks' in features
            # if 'context_masks' in features:
            context_masks = features['context_masks']

            attention_mask = features['attention_mask']
            # if context_masks is not None:
            import torch
            assert len(context_masks) == len(attention_mask)
            n = len(attention_mask)
            # print(n)

            for local_idx in range(n):
                local_trans_features = {'input_ids': features['input_ids'][local_idx].unsqueeze(0),
                                        'attention_mask': features['attention_mask'][local_idx].unsqueeze(0),}
                # print(local_trans_features['input_ids'].shape,local_trans_features['attention_mask'].shape)
                # print(trans_features['input_ids'].shape, trans_features['attention_mask'].shape)
                if context_masks[local_idx].item()==0:
                    # print('use model_without_prompt')
                    output_states = self.model_without_prompt(**local_trans_features, return_dict=False)
                else:
                    # print('use model_with_prompt')
                    output_states = self.model_with_prompt(**local_trans_features, return_dict=False)
                if local_idx==0:
                    output_tokens = output_states[0]
                else:
                    output_tokens = torch.cat([output_tokens, output_states[0]], dim=0)
                # print('output_tokens.shape: ',output_tokens.shape)
            for local_idx in range(n):
                assert torch.sum(attention_mask[local_idx]).item() >= context_masks[local_idx].item(), \
                    f'{attention_mask[local_idx]}, {context_masks[local_idx]}, ' \
                    f'{torch.sum(attention_mask[local_idx]).item()}, {context_masks[local_idx].item()}'
                attention_mask[local_idx][:context_masks[local_idx]] = 0

            # print('forward here')
            features.update({'token_embeddings': output_tokens, 'attention_mask': attention_mask})

            # if self.auto_model.config.output_hidden_states:
            #     all_layer_idx = 2
            #     if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
            #         all_layer_idx = 1
            #
            #     hidden_states = output_states[all_layer_idx]
            #     features.update({'all_layer_embeddings': hidden_states})

            return features
        else:
            context_masks = None
            if 'context_masks' in features:
                context_masks = features['context_masks']
            output_states = self.auto_model(**trans_features, return_dict=False)
            output_tokens = output_states[0]
            attention_mask = features['attention_mask']
            if context_masks is not None:
                import torch
                assert len(context_masks) == len(attention_mask)
                n = len(attention_mask)
                # print('n ',n)
                for local_idx in range(n):
                    assert torch.sum(attention_mask[local_idx]).item() >= context_masks[local_idx].item(),\
                        f'{attention_mask[local_idx]}, {context_masks[local_idx]}, ' \
                        f'{torch.sum(attention_mask[local_idx]).item()}, {context_masks[local_idx].item()}'
                    attention_mask[local_idx][:context_masks[local_idx]] = 0

            # print('forward here')
            features.update({'token_embeddings': output_tokens, 'attention_mask': attention_mask})

            if self.auto_model.config.output_hidden_states:
                all_layer_idx = 2
                if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                    all_layer_idx = 1

                hidden_states = output_states[all_layer_idx]
                features.update({'all_layer_embeddings': hidden_states})

            return features

    def get_word_embedding_dimension(self) -> int:
        if hasattr(self,'auto_model'):
            return self.auto_model.config.hidden_size
        return self.model_with_prompt.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]

            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

            # Lowercase
            if self.do_lower_case:
                to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

            tokenized = self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length)

        # elif isinstance(texts[0], dict):
        #     to_tokenize = []
        #     output['text_keys'] = []
        #     for lookup in texts:
        #         text_key, text = next(iter(lookup.items()))
        #         to_tokenize.append(text)
        #         output['text_keys'].append(text_key)
        #     to_tokenize = [to_tokenize]
        elif isinstance(texts[0], list):
            import torch
            assert isinstance(texts[0][1],str)
            new_texts = []
            for s in texts:
                if self.do_lower_case:
                    new_texts.append([s[0],s[1].strip().lower(),s[2]])
                else:
                    new_texts.append([s[0], s[1].strip(), s[2]])
            texts = new_texts
            if len(texts[0])==3:
                # print('component 3')
                num = len(texts)
                contexts = []
                concatenated_input_texts = []
                for local_idx in range(num):
                    assert len(texts[local_idx])==3
                    contexts.append(texts[local_idx][0])
                    concatenated_input_texts.append(''.join(texts[local_idx][:-1]))
                    assert isinstance(contexts[-1],str)
                    assert isinstance(concatenated_input_texts[-1],str)
                tokenized = self.tokenize(concatenated_input_texts)
                context_tok = self.tokenize(contexts)
                tokenized['context_masks'] = torch.sum(context_tok['attention_mask'],dim=1)
                tokenized['context_masks'] = tokenized['context_masks']-1
                for my_idx in range(len(tokenized['context_masks'])):
                    if tokenized['context_masks'][my_idx]<=1:
                        tokenized['context_masks'][my_idx] = 0
                text_types = [pair[-1] for pair in texts]
                # print(text_types)
                assert all([tid==1 for tid in text_types]) or all([tid==0 for tid in text_types])
                tokenized['text_type'] = text_types[0]
                # torch.set_printoptions(edgeitems=15)
                # print(tokenized)
                # exit(0)
            elif len(texts[0])==2:
                # print('component 2')
                input_texts = [pair[0] for pair in texts]
                text_types = [pair[-1] for pair in texts]
                assert all([tid == 1 for tid in text_types]) or all([tid == 0 for tid in text_types])
                tokenized = self.tokenize(input_texts)
                tokenized['text_type'] = text_types[0]
            else:
                raise ValueError('tokenization error')
        else:
            raise ValueError('not support other modes')
            # batch1, batch2 = [], []
            # for text_tuple in texts:
            #     batch1.append(text_tuple[0])
            #     batch2.append(text_tuple[1])
            # to_tokenize = [batch1, batch2]

        output.update(tokenized)
        return output


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        if self.model_name_or_path == 'bi-contriever':
            print('save bi-contriever')
            print(f'query model save to: {output_path}query')
            print(f'doc model save to: {output_path}doc')
            self.query_model.save_pretrained(f'{output_path}query')
            self.doc_model.save_pretrained(f'{output_path}doc')
            self.tokenizer.save_pretrained(f'{output_path}query')
            self.tokenizer.save_pretrained(f'{output_path}doc')

            with open(os.path.join(f'{output_path}query', 'sentence_bert_config.json'), 'w') as fOut:
                json.dump(self.get_config_dict(), fOut, indent=2)
            with open(os.path.join(f'{output_path}doc', 'sentence_bert_config.json'), 'w') as fOut:
                json.dump(self.get_config_dict(), fOut, indent=2)
        elif self.model_name_or_path.startswith('bigtr'):
            print('save bi-gtr')
            print(f'model_without_prompt save to: {os.path.join(output_path,"without_prompt")}')
            print(f'model_with_prompt save to: {os.path.join(output_path,"with_prompt")}')
            self.model_without_prompt.save_pretrained(os.path.join(output_path,'without_prompt'))
            self.model_with_prompt.save_pretrained(os.path.join(output_path,'with_prompt'))
            self.tokenizer.save_pretrained(os.path.join(output_path,"without_prompt"))
            self.tokenizer.save_pretrained(os.path.join(output_path, "with_prompt"))
            self.tokenizer.save_pretrained(output_path)

            # with open(os.path.join(os.path.join(output_path,"without_prompt"), 'sentence_bert_config.json'), 'w') as fOut:
            #     json.dump(self.get_config_dict(), fOut, indent=2)
            # with open(os.path.join(os.path.join(output_path,"with_prompt"), 'sentence_bert_config.json'), 'w') as fOut:
            #     json.dump(self.get_config_dict(), fOut, indent=2)
            with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
                json.dump(self.get_config_dict(), fOut, indent=2)
        else:
            self.auto_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

            with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
                json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return Transformer(model_name_or_path=input_path, **config)





