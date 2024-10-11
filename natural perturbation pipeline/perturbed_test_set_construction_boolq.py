import json
import pandas as pd
import random

def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

filepath = './dev.jsonl'
json_data = read_jsonl_file(filepath)

def Find_Element_All_Occurrences(example_list, element):
  return [index for index in range(len(example_list)) if example_list[index] == element]

can_ori_passages = pd.read_csv('./boolq.csv')['original passages'].values.tolist()
can_per_passages = pd.read_csv('./boolq.csv')['modified_passages'].values.tolist()
answers, questions, contexts, perturbed_contexts = [], [], [], []
for example_id in range(len(json_data)):
  context = json_data[example_id]['passage']
  question = json_data[example_id]['question']
  answer = json_data[example_id]['answer']

  indices = Find_Element_All_Occurrences(can_ori_passages, context)

  if len(indices) > 0:
    if len(indices) == 1:
      perturbed_text = can_per_passages[indices[0]]
      contexts.append(context)
      questions.append(question)
      answers.append(answer)
      perturbed_contexts.append(perturbed_text)
    else:
      perturbed_text = can_per_passages[random.sample(indices, 1)[0]]
      contexts.append(context)
      questions.append(question)
      answers.append(answer)
      perturbed_contexts.append(perturbed_text)

data = {
    'original context': contexts,
    'perturbed context': perturbed_contexts,
    'question': questions,
    'answer': answers
}

df = pd.DataFrame(data)

csv_filename = './boolq-data.csv'
df.to_csv(csv_filename, index=False)