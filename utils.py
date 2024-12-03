import openai
import re
import json
import os
import torch
import clip
from nltk.corpus import wordnet as wn
from torch.cuda.amp import GradScaler


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class Assistant:
    mero_prompt_template = '''Please provide a list of nouns that describe the meronyms of **{input}**. Ensure the following criteria are met:
1. **Mutually Exclusive**: Each noun should represent a distinct component without overlapping with others.
2. **Non-hypernym**: All nouns should not be hypernyms of Input.
3. **WordNet Noun**: Each noun is from WordNet.
4. **Format**: Present the nouns in a list format as follows: $[noun_1, noun_2, noun_3, ...]$.

**Input with description**: {input}, {desc}
'''
    select_prompt_template = '''You are to select the meronym that best matches '{name} - {definition}'. Here are the options:
{options}
Please select the most appropriate synset by providing only the number corresponding to your choice.'''

    def __init__(self, model: str = "gpt-4-1106-preview", temperature: float = 1.0, state_file: str = 'hierarchy_state.json'):
        """
        Initialize the Assistant object with the OpenAI API key and model name.

        Args:
            model (str): OpenAI model to use (default is "gpt-4-1106-preview").
            temperature (float): Sampling temperature.
            depth (int): Depth of the hierarchy.
        """
        self.client = openai.OpenAI(
            api_key="sk-tzqTKurE3l4Pu5QJE6D12347Ff7b4343842f929168B0Fc5c",
            base_url="https://aihubmix.com/v1"
        )
        self.model = model
        self.temperature = temperature
        self.state_file = state_file

        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("resources/weights/ViT-L-14.pt", device=self.device)

    def chat(self, messages):
        """
        Send a message to the GPT model and return the response.
        """
        try:
            # Call the OpenAI API to get the response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )

            # Extract the assistant's response from the API response
            assistant_message = completion.choices[0].message.content

            # Return the assistant's response
            return assistant_message.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise e

    def get_meronyms(self, labels):
        """
        Get meronyms labels from base labels.

        Args:
            labels (list): A list of base labels.

        Returns:
            dict: A hierarchical dictionary representing labels at each layer.
        """
        # Initialize hierarchy and processed labels
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                hierarchy = state.get('hierarchy', {})
                processed_labels = state.get('processed_labels', [])
        else:
            hierarchy = {}
            processed_labels = []

        remaining_labels = [label for label in labels if label not in processed_labels]
        total_labels = len(remaining_labels)

        for idx, (synset_id, noun) in enumerate(remaining_labels):
            print(f"Processing {noun} ({idx+1+len(labels)-total_labels}/{len(labels)})")
            try:
                # Initialize the hierarchy for this synset
                hierarchy[noun] = []
                # Get meronyms from WordNet
                # meronyms = self.get_meronyms_from_wordnet(synset_id)

                # If number of meronyms is 0, use chat to get more
                # if len(meronym_dict) == 0:
                #     meronyms = self.get_meronyms_via_chat(noun, synset_id, meronyms)

                meronyms = self.get_meronyms_via_chat(noun, synset_id, set())
                # Build the next level in the hierarchy
                for meronym_synset_name in meronyms:
                    hierarchy[noun].append(f"{meronym_synset_name}")

                # Mark this label as processed
                processed_labels.append((synset_id, noun))
                # Save the state after processing each label
                state = {
                    'hierarchy': hierarchy,
                    'processed_labels': processed_labels,
                }
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4)

            except Exception as e:
                print(f"Error processing {noun}: {e}")
                # Save the state before exiting
                state = {
                    'hierarchy': hierarchy,
                    'processed_labels': processed_labels,
                }
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4)
                raise e  # Re-raise the exception to handle it externally

        # Save the final hierarchy
        with open(self.state_file, 'w') as f:
            json.dump(hierarchy, f, indent=4)

        return hierarchy

    def get_meronyms_from_wordnet(self, synset_id):
        synset = wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))
        meronyms = synset.part_meronyms() + synset.substance_meronyms() + synset.member_meronyms()
        return set(meronyms)

    def get_meronyms_via_chat(self, label, label_synset_id, meronym_set):
        try:
            label_synset = wn.synset_from_pos_and_offset(label_synset_id[0], int(label_synset_id[1:]))
            # Build the conversation messages
            assistant_response = self.chat([
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": self.mero_prompt_template.format(input=label, desc=label_synset.definition())}
            ])

            # Parse the assistant's response
            match = re.search(r'\$\[(.*?)\]\$', assistant_response, re.DOTALL)
            content = match.group(1)
            # Split the content by commas to get the list of nouns
            additional_meronyms = [noun.strip().strip('\'"') for noun in content.split(',')]

            for meronym in additional_meronyms:
                meronym_synset_name = self.get_related_meronym_synset_name(meronym, label_synset)
                if meronym_synset_name not in meronym_set:
                    meronym_set.add(meronym_synset_name)
            return list(meronym_set)
        except Exception as e:
            print(f"Error during chat for {label}: {e}")
            raise e

    def get_related_meronym_synset_name(self, meronym, label_synset):
        meronym_synsets = wn.synsets(meronym, pos=wn.NOUN)
        if not meronym_synsets:
            print(f"No synsets found for meronym '{meronym}'.")
            return meronym

        options = []
        for idx, syn in enumerate(meronym_synsets):
            options.append(f"{idx + 1}. Name: {syn.name().split('.')[0]}, Definition: {syn.definition()}")

        while True:
            # Call the chat function with the designed prompt
            assistant_response = self.chat([
                {"role": "system", "content": 'You are a helpful assistant.'},
                {"role": "user", "content": self.select_prompt_template.format(name=label_synset.name(), definition=label_synset.definition(), options=chr(10).join(options))}
            ])

            # Extract the number from the assistant's response
            selected_number = re.findall(r'\d+', assistant_response)
            if not selected_number:
                continue

            selection = int(selected_number[0])
            if 1 <= selection <= len(meronym_synsets):
                meronym_synset = meronym_synsets[selection - 1]
                break

        # If all checks pass, return the synset name
        return meronym_synset.name()


def meronyms_with_definition(mero_label_to_idx):
    meronyms = []
    for label, _ in mero_label_to_idx.items():
        if len(label.split('.')) > 1:
            synset = wn.synset(label)
            meronyms.append(f"{label.split('.')[0]}: {synset.definition()}")
        else:
            meronyms.append(label)

    background_synsets = wn.synsets("background", pos=wn.NOUN)[0]
    meronyms.append(f"background: {background_synsets.definition()}")
    return meronyms


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt
