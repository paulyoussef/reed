
import torch
from transformers import set_seed
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTJForCausalLM

import json
import random
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset

import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import entropy


import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns

def classify(probs1, probs2, ent=False):
    # Example data
    list1 = np.array([x[:10] for x in probs1]) # [[1.1, 1.2], [1.2, 1.1], [1.3, 1.4], [1.4, 1.3]]  # Class B instances
    list2 = np.array([x[:10] for x in probs2]) # [[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3]]  # Class A instances

    if ent:
        list1 = np.array([entropy(x) for x in list1]).reshape((-1,1))
        list2 = np.array([entropy(x) for x in list2]).reshape((-1,1))

    # Split each class data into training and testing sets
    list1_train, list1_test = train_test_split(list1, test_size=0.5, shuffle=False)
    list2_train, list2_test = train_test_split(list2, test_size=0.5, shuffle=False)

    # Combine the training data and create labels
    X_train = np.vstack((list1_train, list2_train))
    y_train = np.hstack((np.zeros(len(list1_train)), np.ones(len(list2_train))))

    # Combine the testing data and create labels
    X_test = np.vstack((list1_test, list2_test))
    y_test = np.hstack((np.zeros(len(list1_test)), np.ones(len(list2_test))))


    shuffled_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]



    # Create and train the logistic regression model with L1 regularization
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')

    print(report)
    return list1, list2, np.round(accuracy*100,2)

def get_logits(model, tokenizer, inputs, device):
    inputs_tokenized = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs_tokenized).logits # shape: batch_size x num tokens x vocab size
    last_non_masked = inputs_tokenized["attention_mask"].sum(1) - 1 # index of the last non-padding token
    to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
    gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size 

    return torch.nn.functional.log_softmax(gathered, dim=1)   

def get_logits_grad(model, tokenizer, inputs, device):
    inputs_tokenized = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    logits = model(**inputs_tokenized).logits # shape: batch_size x num tokens x vocab size
    last_non_masked = inputs_tokenized["attention_mask"].sum(1) - 1 # index of the last non-padding token
    to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
    gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size 

    return torch.nn.functional.log_softmax(gathered, dim=1)   

def get_logits_input_embedds(model, tokenizer, inputs_tokenized, input_embeds, device):
    #inputs_tokenized = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)

    logits = model(inputs_embeds= input_embeds).logits
    last_non_masked = inputs_tokenized["attention_mask"].sum(1) - 1 # index of the last non-padding token
    to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
    gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size 

    return torch.nn.functional.log_softmax(gathered, dim=1)   


def get_logits_batched(model, tokenizer, inputs, device):
    dl = DataLoader(inputs, shuffle=False, batch_size= 4)
    tensor_list = []
    for batch in dl: 
        inputs_tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)

        with torch.no_grad():
            logits = model(**inputs_tokenized).logits # shape: batch_size x num tokens x vocab size
        last_non_masked = inputs_tokenized["attention_mask"].sum(1) - 1 # index of the last non-padding token
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
        gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size 
        tensor_list.append(torch.nn.functional.log_softmax(gathered, dim=1).detach().cpu() )
    return  torch.cat(tensor_list, dim=0)


class CustomDataset(Dataset):
    def __init__(self, normal, tuned):
        self.normal = normal
        self.tuned = tuned
        assert len(normal) == len(tuned)
    
    def __len__(self):
        return len(self.normal)
    
    def __getitem__(self, idx):
        return self.normal[idx], self.tuned[idx]
    

# The indices of nearest neighbours are stored in corpus_idx.txt.
with open('../corpus_idx.txt', 'r') as fIn:
    lines = fIn.readlines()
    lines = [line[:-1] for line in lines]
    corpus_idx = [[int(idx) for idx in line.split()] for line in lines]


def get_probs(model, tokenizer, inputs, device, bs = 32):
    data_loader = DataLoader(inputs, batch_size = bs, shuffle= False)
    model.eval()
    results = []
    answers = []
    for b in data_loader:
        batch = tokenizer(b, return_tensors='pt', padding=True, truncation=True)
        batch.to(device)
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'], output_hidden_states=True)

        if tokenizer.padding_side == 'right': 
            last_index = batch['attention_mask'].sum(axis=1) - 1
            probs =  torch.nn.functional.softmax(outputs.logits[torch.arange(batch['input_ids'].size(0)), last_index], dim=1)

            ## get preds
            logits = outputs.logits
            last_non_masked = batch["attention_mask"].sum(1) - 1 # index of the last non-padding token
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
            gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size
            ans = torch.argmax(gathered, dim=1) # shape: batch_size
        
            answers += ans.detach().cpu().numpy().tolist()

        elif tokenizer.padding_side == 'left':
            raise ValueError # probs = torch.nn.functional.softmax(outputs.logits[torch.arange(batch['input_ids'].size(0)), torch.empty(batch['input_ids'].size(0), dtype=torch.int32).fill_(-1)])
        
        reprs_b = torch.topk(probs, 100).values
        results.append(reprs_b.detach().cpu().numpy() )
    results = np.concatenate(results, axis=0)
    return results, np.array(answers)


def construct_icl_examples(idx, demos, clean=False):
    # len: 32
    order = [2, 1, 2, 0, 1, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    random.shuffle(order)
    icl_examples = []
    demo_ids = corpus_idx[idx]
    demo_ids = demo_ids[:len(order)]
    for demo_id, o in zip(demo_ids, order):
        line = demos[demo_id-2000]
        new_fact = line['requested_rewrite']['prompt'].format(line['requested_rewrite']['subject'])
        target_new = line['requested_rewrite']['target_new']['str']
        target_true = line['requested_rewrite']['target_true']['str']

        if not clean:
            if o == 0:
                # same prompt for "updating" and querying, both use taret_new
                # example: New Fact: The mother tongue of Robert Lecourt is English\nPrompt: The mother tongue of Robert Lecourt is English
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_new}\n\n')
            elif o == 1:
                # one prompt for "updating" and another prompt for querying, both use taret_new
                # example: New Fact: The mother tongue of Colette Darfeuil is Russian\nPrompt: Colette Darfeuil spoke the language Russian
                prompt = random.choice(line['paraphrase_prompts'])
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_new}\n\n')
            elif o == 2:
                # one prompt with target_new, another prompt with target_true
                # example: New Fact: The mother tongue of Marc-Philippe Daubresse is Russian\nPrompt: The mother tongue of Melchior de Vogüé is French
                prompt = random.choice(line['neighborhood_prompts'])
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
        else:
            # clean setting : teach model to ignore "New Fact"
            if o == 0:
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {new_fact} {target_true}\n\n')
            elif o == 1:
                prompt = random.choice(line['paraphrase_prompts'])
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')
            elif o == 2:
                prompt = random.choice(line['neighborhood_prompts'])
                icl_examples.append(f'New Fact: {new_fact} {target_new}\nPrompt: {prompt} {target_true}\n\n')

    icl_examples.reverse()
    return icl_examples



def get_match_acc(model, tok, inputs1, inputs2, device, bs = 32):
    #assert inputs1[0] != inputs2[0]

    preds1 = get_preds(model, tok, inputs1, device, bs = bs)
    preds2 = get_preds(model, tok, inputs2, device, bs = bs)
    
    return np.mean(preds1 == preds2)


def get_preds(model, tok, inputs, device, bs=32):
    answers = []

    data_loader = DataLoader(inputs, batch_size = bs, shuffle= False)

    for prompts in data_loader: 

        prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt").to(device)
           
        with torch.no_grad():
            logits = model(**prompt_tok).logits # shape: batch_size x num tokens x vocab size
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1 # index of the last non-padding token
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1) # shape: batch_size x 1 x vocab size
            gathered = torch.gather(logits, 1, to_gather).squeeze(1) # shape: batch_size x vocab_size
            ans = torch.argmax(gathered, dim=1) # shape: batch_size
        
            answers += ans.detach().cpu().numpy().tolist()

    assert len(answers) == len(inputs)
    return np.array(answers)



def get_bos_topk(frozen_model, model_name, tokenizer, k):
    # indices of closest tokens to BOS
    if 'vicuna' in model_name.lower() or 'llama' in model_name.lower():
        e_mat = frozen_model.model.embed_tokens.weight
    else:
        e_mat = frozen_model.transformer.wte.weight

    bos_embedding = e_mat[tokenizer.bos_token_id].unsqueeze(0)
    sim_list = torch.nn.functional.cosine_similarity(e_mat, bos_embedding).cpu().detach().numpy()

    # Sample list of float numbers
    float_list = sim_list.tolist()

    # Step 1: Pair each element with its index
    indexed_list = list(enumerate(float_list))

    # Step 2: Sort the pairs based on the elements in descending order
    sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)

    # Step 3: Extract the indices from the sorted pairs
    sorted_indices = [(index,value) for index, value in sorted_indexed_list]
    ids = [i for i,_ in sorted_indices]
    return ids[:k]

def run(model_name, seed, epochs, rt, discrete, lweight, repeat_factor, ds_name):

    set_seed(seed)
    assert not discrete
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    RESULTS_DIR = './context-dims/' + ds_name


    RESULTS_DIR += f'/{seed}/'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
                
    assert lweight != 0.

    if 'gpt-j' in model_name:
        frozen_model = GPTJForCausalLM.from_pretrained(model_name).to(device)
    elif 'gpt2' in model_name:
        frozen_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    elif 'vicuna' in model_name.lower() or 'llama' in model_name.lower():
        from transformers import LlamaForCausalLM 
        frozen_model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    else:
        raise ValueError

    frozen_model.eval()
    for _, p in frozen_model.named_parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO reversal tokens 
    reversal_tokens = ['[reversal_token_{}]'.format(t+1) for t in range(rt)]*repeat_factor
    
    #tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left'

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenizer.add_tokens(reversal_tokens)
    frozen_model.resize_token_embeddings(len(tokenizer))

    # store initial values for reversal tokens
    model_name_split = model_name.split('/')[1]
    natural_tokens_ids = []
    for t in reversal_tokens:
        if 'gpt' in model_name:
            high_threshold = frozen_model.transformer.wte.weight.shape[0]
        elif 'llama' in model_name.lower():
            high_threshold = frozen_model.model.embed_tokens.weight.shape[0]

        rand_int = np.random.randint(0, high_threshold)
        assert rand_int != tokenizer.bos_token_id
        natural_tokens_ids.append(rand_int)
        if 'gpt' in model_name:
            frozen_model.transformer.wte.weight[tokenizer(t, add_special_tokens=False)['input_ids']].data.copy_(frozen_model.transformer.wte.weight[rand_int])
        elif 'llama' in model_name.lower():
            frozen_model.model.embed_tokens.weight[tokenizer(t, add_special_tokens=False)['input_ids']].data.copy_(frozen_model.model.embed_tokens.weight[rand_int] )


    if not discrete:
        for t in reversal_tokens:
            if 'gpt' in model_name:
                tuned_rt = frozen_model.transformer.wte.weight[tokenizer(t, add_special_tokens=False)['input_ids']][0]
            elif 'llama' in model_name.lower():
                tuned_rt = frozen_model.model.embed_tokens.weight[tokenizer(t, add_special_tokens=False)['input_ids']][0]

            np.savetxt(f'{RESULTS_DIR}{model_name_split}_epochs_{epochs}_rt_{rt}_rf_{repeat_factor}_{t}_lambda_{np.round(lweight,2)}_init.txt', tuned_rt.detach().cpu().numpy(), fmt='%f')

    lines = []
    ike_inputs = []
    ike_inputs_clean = []
    normal_inputs = []
    ike_inputs_bos = []

    joined_reversal_tokens = ' '.join(reversal_tokens)
    

    if ds_name == 'counterfact':
        with open('../counterfact.json', 'r') as f:
            lines = json.load(f)
            
        icl_examples = []
        demos = lines[2000:]
        lines = lines[:2000]


        # icl_cnt = 0
        example_idx = 0

   

        for i, line in enumerate(lines):

            #if i % 10 == 0:
            #    print(i, success_cnt, total_cnt, magnitude / (total_cnt + 1e-12), para_success_cnt, para_magnitude / (para_total_cnt + 1e-12), orig_success_cnt ,orig_magnitude / (i + 1e-12))
            relation = line['requested_rewrite']['relation_id'] # e.g., P103
            prompt = line['requested_rewrite']['prompt'] # e.g., 'The mother tongue of Danielle Darrieux is'
            subject = line['requested_rewrite']['subject'] # 'Danielle Darrieux'
            prompt_calibrate = prompt.format('SUBJECT') # The mother tongue of SUBJECT is
            prompt = prompt.format(subject) # 'The mother tongue of Danielle Darrieux is'
            PROMPTS = [prompt, prompt_calibrate] # ['The mother tongue of Danielle Darrieux is', 'The mother tongue of SUBJECT is']

            target_true = line['requested_rewrite']['target_true']['str'] # French
            target_new = line['requested_rewrite']['target_new']['str'] # English
            
            PPLs = []
            targets = [target_new, target_true]
            icl_examples = construct_icl_examples(example_idx, demos)
            icl_examples_clean = construct_icl_examples(example_idx, demos, clean=True)

            # 'New Fact: The mother tongue of Danielle Darrieux is English\nPrompt: The mother tongue of Danielle Darrieux is English\n\n'
            icl_examples.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_new}\n\n')
            icl_examples_clean.append(f'New Fact: {prompt} {target_new}\nPrompt: {prompt} {target_true}\n\n')

            example_idx += 1

            ike_inputs.append(''.join(icl_examples) + f'{prompt}')
            ike_inputs_bos.append(''.join(icl_examples) + f' {joined_reversal_tokens} {prompt}')

            ike_inputs_clean.append(''.join(icl_examples_clean) + f'{prompt}')

            normal_inputs.append( f'{prompt}')
    elif ds_name == 'mquake':
        with open('../MQuAKE-CF-3k-v2.json', 'r') as f:
            lines = json.load(f)

        np.random.shuffle(lines)
        for i in range(2000):
            prompt = lines[i]['questions'][0]
            icl_prompt = []
            for x in lines[i]['requested_rewrite']:
                icl_prompt.append(x['question'] + ' ' + x['target_new']['str'])

            icl_prompt += [prompt + ' ' + lines[i]['new_answer']]

            ike_inputs.append('\n'.join(icl_prompt) + f' {prompt}')
            # ike prompt + reversal tokens +  query
            ike_inputs_bos.append('\n'.join(icl_prompt) + f' {joined_reversal_tokens} {prompt}')
            #normal_bos_inputs.append( f'{joined_reversal_tokens} {prompt}')
            # only query
            normal_inputs.append( f'{prompt}')

    else:
        raise ValueError
    if 'gpt' in model_name:
        var = frozen_model.transformer.wte.weight # [tokenizer(tokenizer.bos_token)['input_ids']]
        frozen_model.transformer.wte.weight.requires_grad = True
    elif 'llama' in model_name.lower():
        var = frozen_model.model.embed_tokens.weight
        frozen_model.model.embed_tokens.weight.requires_grad = True

    normal_train, normal_test, bos_train, bos_test= train_test_split(normal_inputs, ike_inputs_bos, test_size=0.5, random_state=1)
    normal_train, normal_dev, bos_train, bos_dev= train_test_split(normal_train, bos_train, test_size=0.1, random_state=2)

    print(len(normal_train), len(bos_train))
    print(len(normal_dev), len(bos_dev))
    print(len(normal_test), len(bos_test))

    if 'gpt2' in model_name:
        inf_bs = 16
    else:
        inf_bs = 8

    example = tokenizer(bos_test[:2], return_tensors='pt', padding=True, truncation=True)['input_ids'][0]
    for x in reversal_tokens:
        tmp = tokenizer(x, add_special_tokens=False)['input_ids'][0]
        assert tmp in example

    # Get probs/preds before adapting embeddings
    ike_bos_untrained = get_probs(frozen_model, tokenizer, bos_test, device, bs = inf_bs)
    normal_untrained = get_probs(frozen_model, tokenizer, normal_test, device, bs = inf_bs) # on the untrained model ? shouldn't matter, since no BOS


    if 'gpt2' in model_name:
        bs = 4
    elif 'gpt-j' in model_name or 'llama' in model_name.lower():
        bs = 2


    dataset = CustomDataset(normal_train, bos_train)

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    dist_weight = torch.nn.Parameter(torch.tensor(lweight, requires_grad=False))
    train_params = [var]

    optimizer = torch.optim.Adam(train_params)
    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    # indices of reversal tokens
    selected_indices = torch.tensor([x[0] for x in tokenizer(reversal_tokens, add_special_tokens=False)['input_ids']], device=device)
    avg_other = var[natural_tokens_ids].mean(dim=0).detach().unsqueeze(0) 
    criterion = torch.nn.CosineEmbeddingLoss()
    target_cos_loss = torch.ones(1, device=device)

    # some weights before training
    before_const = var[0].sum().cpu().detach().clone().numpy()
    var_index = tokenizer(reversal_tokens[0], add_special_tokens=False)['input_ids'][0]
    before_var = var[var_index].sum().cpu().detach().clone().numpy()

    dev_loss_vals = []
    for i in range(epochs):
        print(i)
        for train_step, (normal_batch, bos_batch) in enumerate(dataloader):

            normal_logits = get_logits(frozen_model, tokenizer, normal_batch, device)
            normal_logits = normal_logits.detach()

            optimizer.zero_grad()
            
            ike_logits = get_logits_grad(frozen_model, tokenizer, bos_batch, device)

            avg_rt = var[selected_indices].mean(dim=0)

            output =   (1 - torch.abs(dist_weight))*loss(ike_logits, normal_logits) + torch.abs(dist_weight)*criterion(avg_rt.unsqueeze(0), avg_other, target_cos_loss)

        
            output.backward()

            mask = torch.zeros_like(var.grad)

            for trainable_index in tokenizer(reversal_tokens, add_special_tokens=False)['input_ids']:
                mask[trainable_index[0], :] = 1
            # Multiply the original tensor with the mask
            var.grad = var.grad * mask

            optimizer.step()

    # same weights after training
    after_const = var[0].sum().cpu().detach().clone().numpy()
    after_var = var[var_index].sum().cpu().detach().clone().numpy()
    
    assert before_var != after_var
    assert before_const == after_const

    ike_bos = get_probs(frozen_model, tokenizer, bos_test, device, bs = inf_bs)
    normal = get_probs(frozen_model, tokenizer, normal_test, device, bs = inf_bs) # on the untrained model ? shouldn't matter, since no BOS

    pairs = [
    
        (normal_untrained, ike_bos_untrained, 'unedited vs. IKE ',
          'unedited', 'IKE-edited'), # normal vs. bos: IKE-edited w/ BOS vs. normal

        (normal_untrained, ike_bos, 'unedited vs. tuned',
          'unedited', 'IKE-edited + tuned'), # normal vs. bos: IKE-edited w/ BOS vs. normal

       (normal_untrained, normal, 'unedited vs. unedited (after tuning)',
          'unedited', 'unedited (after tuning)'), # Sanity check

        ]        

    entropy_options = [False]

    for c1, c2, setting, s1, s2 in pairs: 
        
        p1, preds1 = c1[0], c1[1]
        p2, preds2 = c2[0], c2[1]

        for ent in entropy_options:
            list1, list2, accuracy = classify(p1, p2, ent=ent)

            same_preds = np.round(np.mean(preds1 == preds2)*100,2)

            with open(f'{RESULTS_DIR}results_tuned.csv', 'a') as f:
                f.write('{},{},{},{},{},{},{},{:.2f},{:.2f}\n'.format(model_name, setting, str(ent), epochs, rt, repeat_factor, np.round(dist_weight.item(), 2), accuracy, same_preds))


            dict = {
            'Probs':np.mean(list1, axis=0).tolist() +  np.mean(list2, axis=0).tolist(),
            'Prompts' : [s1]*10 + [s2]*10
            }

            if ent:
                continue

            df = pd.DataFrame.from_dict(dict)

            plt.figure(figsize=(8,5), dpi=320)
            plt.rcParams.update({'font.size': 16})
            sns.kdeplot( x='Probs', hue='Prompts', data=df, palette="Set2", fill=True, bw_adjust=0.6, cumulative=False)

            plt.savefig(f'{RESULTS_DIR}{setting}-rt-{rt}-rf-{repeat_factor}-epochs-{epochs}-lambda-{np.round(dist_weight.item(),2)}' + model_name.split('/')[-1] + '.pdf', format='pdf')

            # Show the plot
            plt.show()


    # store loss


    with open(f'{RESULTS_DIR}loss_vals_{model_name_split}_epochs_{epochs}_rt_{rt}_rf_{repeat_factor}_lambda_{np.round(dist_weight.item(),2)}.csv', 'w') as f:
        for v in dev_loss_vals:
            f.write(str(v) + '\n')

    for t in reversal_tokens:
        if 'gpt' in model_name:
            tuned_rt = frozen_model.transformer.wte.weight[tokenizer(t, add_special_tokens=False)['input_ids']][0]
        elif 'llama' in model_name.lower():
            tuned_rt = frozen_model.model.embed_tokens.weight[tokenizer(t, add_special_tokens=False)['input_ids']][0]

        np.savetxt(f'{RESULTS_DIR}{model_name_split}_epochs_{epochs}_rt_{rt}_rf_{repeat_factor}_{t}_lambda_{np.round(dist_weight.item(),2)}.txt', tuned_rt.detach().cpu().numpy(), fmt='%f')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model", required=False, type = str, default='gpt2-xl', help="e.g., 'gpt2-xl'")
    parser.add_argument("--seed", required=False, type = int, default=1)
    parser.add_argument("--epochs", required=True, type = int, default=5)
    parser.add_argument("--rt", required=True, type = int, default=1)
    parser.add_argument("--lweight", required=True, type = float, default=0.5)
    parser.add_argument("--rf", required=True, type = int, default=1)
    parser.add_argument('--discrete', action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", required=True, type = str, default= 'mquake')

    args=parser.parse_args()
    run(args.model, args.seed, args.epochs, args.rt, args.discrete, args.lweight, args.rf, args.dataset)