import subprocess

subprocess.run(["pip", "install", "openai"])

import os
import openai
openai.api_key = key
import time

import json

json_file = open('/content/drive/MyDrive/new datasets/drop_dataset_dev.json', 'r')
data = json.load(json_file)
json_file.close()

contexts = []

for key, value in data.items():
  contexts.append(value['passage'])

instructions, answer = [], []

template = '''Given a reading paragraph, return the Wikipedia page title from which it is likely retrieved.'''

for index in range(len(contexts)):
  instructions.append(template+"\n\n"+'''"""'''+contexts[index]+'''"""''')

for ins_id in range(len(instructions)):
  try:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": instructions[ins_id]}
        ]
    )
    print(response.choices[0].message.content)
    answer.append(response.choices[0].message.content)
    time.sleep(3)

  except openai.error.ServiceUnavailableError:
    print("Service Unavailable. Retrying in 2 minutes...")
    time.sleep(120)  # Wait for 2 minutes before retrying

page_titles = []

import re

def extract_quoted_content(text):
    pattern = r'"(.*?)"'

    matches = re.findall(pattern, text)

    result = ' '.join(matches)

    result = result.rstrip('.')

    return result

for ans in answer:
  page_titles.append(extract_quoted_content(ans))

page_titles_processed = [p for p in page_titles if len(p)>0]

import json
import pandas as pd
import random
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

json_file = open('./drop_dataset_dev.json', 'r')
data = json.load(json_file)
json_file.close()

def Find_Element_All_Occurrences(example_list, element):
  return [index for index in range(len(example_list)) if example_list[index] == element]

can_ori_passages = pd.read_csv('./drop.csv')['original passages'].values.tolist()
can_per_passages = pd.read_csv('./drop.csv')['modified_passages'].values.tolist()

contexts, questions, answers, perturbed_contexts = [], [], [], []

for key, value in data.items():
  tem_context, tem_perturbed_context = [], []

  joined_passage = value['passage']
  sentences = sent_tokenize(joined_passage.strip())
  context = ' '.join(sentences)

  contexts.append(context)
  tem_context.append(context)
  indices = Find_Element_All_Occurrences(can_ori_passages, context)
  if len(indices) > 0:
    if len(indices) == 1:
      perturbed_contexts.append(can_per_passages[indices[0]])
      tem_perturbed_context.append(can_per_passages[indices[0]])
    else:
      perturbed_contexts.append(can_per_passages[random.sample(indices, 1)[0]])
      tem_perturbed_context.append(can_per_passages[random.sample(indices, 1)[0]])
  else:
    perturbed_contexts.append(context)
    tem_perturbed_context.append(context)

  qa_pairs = value['qa_pairs']
  for index in range(len(qa_pairs)):
    questions.append(qa_pairs[index]['question'])
    answer = qa_pairs[index]['answer']
    if len(answer['number']) > 0:
      answers.append(answer['number'])
    elif len(answer['spans']) > 0:
      answers.append(answer['spans'])
    else:
      date = answer['date']
      day = date['day']
      month = date['month']
      year = date['year']
      combined_string = ''.join([s for s in [day, month, year] if len(s) > 0])
      answers.append(combined_string)
  contexts.extend(tem_context*(len(qa_pairs)-1))
  perturbed_contexts.extend(tem_perturbed_context*(len(qa_pairs)-1))

data = {
    'original context': contexts,
    'perturbed context': perturbed_contexts,
    'question': questions,
    'answer': answers
}

df = pd.DataFrame(data)

csv_filename = './drop-data.csv'
df.to_csv(csv_filename, index=False)