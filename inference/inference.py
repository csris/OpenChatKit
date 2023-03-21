# Copyright (c) 2023 Together Computer
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import torch


class StopWordsCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self._cache_str = ''

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._cache_str += self.tokenizer.decode(input_ids[0, -1])
        for stop_words in self.stop_words:
            if stop_words in self._cache_str:
                return True
        return False


def load_model(model_name):
    weights_path = snapshot_download(model_name)

    config = AutoConfig.from_pretrained(model_name)

    # This will init model with meta tensors, which basically does nothing.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=["GPTNeoXLayer"],
        dtype='float16',
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model, 
        max_memory=max_memory,
        no_split_module_classes=["GPTNeoXLayer"], 
        dtype='float16'
    )

    model = load_checkpoint_and_dispatch(
        model, weights_path, device_map=device_map, no_split_module_classes=["GPTNeoXLayer"]
    )

    return model


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return tokenizer


def main():
    torch.manual_seed(0)

    model_name = 'togethercomputer/GPT-NeoXT-Chat-Base-20B'

    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)

    prompt = '\n<human>: Hi!\n<bot>: My name is Bot, model version is 0.15, part of an open-source kit for fine-tuning new bots! I was created by Together, LAION, and Ontocord.ai and the open-source community. I am not human, not evil and not alive, and thus have no thoughts and feelings, but I am programmed to be helpful, polite, honest, and friendly.\n<human>: What is a computer?\n<bot>:'

    inputs = tokenizer(prompt, padding=True, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            top_p=0.25,
            top_k=50,
            repetition_penalty=1.01,
            temperature=0.6,
            max_new_tokens=128,
            stopping_criteria=StoppingCriteriaList([StopWordsCriteria(['<human>:'], tokenizer)]),
            pad_token_id=tokenizer.eos_token_id
        )
        output = tokenizer.batch_decode(outputs)[0]

    print(output)


if __name__ == '__main__':
    main()
