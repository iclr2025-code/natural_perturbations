import json

json_file = open('./hotpot_dev_distractor_v1.json', 'r')
data = json.load(json_file)
json_file.close()

answers, questions, contextss = [], [], []

for example_id in range(len(data)):
  answers.append(data[example_id]['answer'])
  questions.append(data[example_id]['question'])

  contexts = data[example_id]['context']
  tem = []
  for context_id in range(len(contexts)):
    title = contexts[context_id][0]
    text = ' '.join([sentence.strip() for sentence in contexts[context_id][1]])
    s = 'Paragraph: '+str(title)+'\n'+str(text)
    tem.append(s)
  contextss.append('\n\n'.join(tem)[:-16])

import pandas as pd
import random
perturbed_contextss = []

statistics = []

def Find_Element_All_Occurrences(example_list, element):
  return [index for index in range(len(example_list)) if example_list[index] == element]

can_ori_passages = pd.read_csv('./hotpotqa.csv')['original passages'].values.tolist()
can_per_passages = pd.read_csv('./hotpotqa.csv')['modified_passages'].values.tolist()

for example_id in range(len(data)):
  contexts = data[example_id]['context']
  tem = []
  for context_id in range(len(contexts)):
    title = contexts[context_id][0]
    text = ' '.join([sentence.strip() for sentence in contexts[context_id][1]])
    indices = Find_Element_All_Occurrences(can_ori_passages, text)
    if len(indices) > 0:
      statistics.append('yes')
      if len(indices) == 1:
        perturbed_text = can_per_passages[indices[0]]
      else:
        perturbed_text = can_per_passages[random.sample(indices, 1)[0]]
    else:
      perturbed_text = text
    s = 'Paragraph: '+str(title)+'\n'+str(perturbed_text)
    tem.append(s)
  perturbed_contextss.append('\n\n'.join(tem)[:-16])

data = {
    'original context': contextss,
    'perturbed context': perturbed_contextss,
    'question': questions,
    'answer': answers
}

df = pd.DataFrame(data)

csv_filename = 'hotpotqa-data.csv'
df.to_csv(csv_filename, index=False)