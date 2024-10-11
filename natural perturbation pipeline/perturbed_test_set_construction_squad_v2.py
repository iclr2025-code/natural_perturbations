import pandas as pd
import random
import json
import tqdm
import re
import math

def Find_Element_All_Occurrences(example_list, element):
  return [index for index in range(len(example_list)) if example_list[index] == element]

def remove_duplicates_strategy(original_list, modified_list):
  ori_one_time = [item for item in original_list if original_list.count(item) == 1]
  if len(ori_one_time) > 0:
    mod_one_time = [modified_list[index] for index in [original_list.index(item) for item in ori_one_time]]
  else:
    mod_one_time = []

  ori_two_times = list(set([item for item in original_list if original_list.count(item) > 1]))
  if len(ori_two_times) > 0:
    mod_two_times = []
    for item in ori_two_times:
      indis = Find_Element_All_Occurrences(original_list, item)
      mod_two_times.append(modified_list[random.sample(indis, 1)[0]])
  else:
    mod_two_times = []

  return [ori_one_time+ori_two_times, mod_one_time+mod_two_times]

def extract_information(file_path):
  df = pd.read_csv(file_path)

  original_passages, modified_passages = [], []

  titles = df['title'].values.tolist()
  indices = [index for index in range(len(titles)) if isinstance(titles[index], str)]
  t = [titles[t_index] for t_index in indices]

  for id in range(len(indices)):
    if id == len(indices) - 1:
      df_mini = df.iloc[indices[id]:]
    else:
      df_mini = df.iloc[indices[id]:indices[id+1]]

    opassages = df_mini['original passages'].values.tolist()
    mpassages = df_mini['modified_passages'].values.tolist()

    original_passages.append(remove_duplicates_strategy(opassages, mpassages)[0])
    modified_passages.append(remove_duplicates_strategy(opassages, mpassages)[1])

  if [len(i) for i in original_passages] == [len(j) for j in modified_passages]:
    print("After processing, same number of passages per article!")

  print(t)
  print([len(i) for i in original_passages])
  print([len(i) for i in modified_passages])

  return [t, original_passages, modified_passages]

results = extract_information('../data.csv') # this is the csv file containing the extracted candidate original and perturbed passage paris for squad 2.0

def Load_Read_SQuAD_Json_File(file_path):
  json_file = open(file_path, 'r')
  data = json.load(json_file)
  json_file.close()

  titles = []

  contexts = []

  for title_id in range(len(data['data'])):
    titles.append(data['data'][title_id]['title'])
    paragraphs = data['data'][title_id]['paragraphs']
    contexts_for_each_title = []
    for paragraph in paragraphs:
      context = paragraph['context']
      contexts_for_each_title.append(context)

    contexts.append(contexts_for_each_title)
  return [titles, contexts]

original = Load_Read_SQuAD_Json_File('../SQuAD/dev-v2.0.json')

common_titles = [title for title in original[0] if title in results[0]]

contexts_v2 = [original[1][index] for index in [original[0].index(title) for title in common_titles]]
contexts_ori = [results[1][index] for index in [results[0].index(title) for title in common_titles]]
contexts_modi = [results[2][index] for index in [results[0].index(title) for title in common_titles]]

common_items = []

for id in range(len(common_titles)):
  common_items.append(list(set(contexts_v2[id]) & set(contexts_ori[id])))

not_match = [index for index in range(len(common_items)) if len(common_items[index]) == 0]

titles_pro = [common_titles[index] for index in range(len(common_titles)) if index not in not_match]

contexts_pro = [common_items[index] for index in range(len(common_items)) if index not in not_match]

contexts_keep = []

for item in contexts_pro:
  for sub_item in item:
    contexts_keep.append(sub_item)

def Load_SQuAD_First_Version_Dev_Set(file_path):
  squad_first_version_dev_set = open(file_path, 'r')
  data = json.load(squad_first_version_dev_set)
  squad_first_version_dev_set.close()
  return data

SQuAD_First_Version_Dev_Set = Load_SQuAD_First_Version_Dev_Set('../SQuAD/dev-v2.0.json')

for title_id in tqdm.tqdm(range(len(SQuAD_First_Version_Dev_Set['data']))):
  SQuAD_First_Version_Dev_Set['data'] = [item for item in SQuAD_First_Version_Dev_Set['data'] if item['title'] in titles_pro]

for title_id in tqdm.tqdm(range(len(SQuAD_First_Version_Dev_Set['data']))):
  paragraphs_squad_first_version_dev_set = SQuAD_First_Version_Dev_Set['data'][title_id]['paragraphs']
  SQuAD_First_Version_Dev_Set['data'][title_id]['paragraphs'] = [item for item in paragraphs_squad_first_version_dev_set if item['context'] in contexts_keep]

with open('./dev-v2.0-original-pre.json', 'w') as f:
  json.dump(SQuAD_First_Version_Dev_Set, f)

contexts_ori_pro = [contexts_ori[index] for index in range(len(contexts_ori)) if index not in not_match]
contexts_modi_pro = [contexts_modi[index] for index in range(len(contexts_modi)) if index not in not_match]

modified_contexts_keep = []

for findex in range(len(contexts_pro)):
  common = contexts_pro[findex]
  temp = []
  for sub in common:
    temp.append(contexts_ori_pro[findex].index(sub))
  for t in temp:
    modified_contexts_keep.append(contexts_modi_pro[findex][t])

def find_continuous_span_start(main_string, substring):
  pattern = re.compile(r'\b' + re.escape(substring) + r'\b')
  match = pattern.search(main_string)
  if match:
    return match.start()
  else:
    return -1

def Load_json_data(file_path):
  file = open(file_path, 'r')
  data = json.load(file)
  file.close()
  return data

d = Load_json_data('./dev-v2.0-original-pre.json')

for title_id in tqdm.tqdm(range(len(d['data']))):
  paragraphs = d['data'][title_id]['paragraphs']
  for paragraph in paragraphs:
    context = paragraph['context']
    context_index = contexts_keep.index(context)
    paragraph['context'] = modified_contexts_keep[context_index]

    qass = paragraph['qas']
    for qas in qass:
      question = qas['question']
      if str(qas['is_impossible']) == "False":
        answers = qas['answers']
        for answer in answers:
          ground_truth_answer = answer['text']
          answer['answer_start'] = find_continuous_span_start(modified_contexts_keep[context_index], ground_truth_answer)
      else:
        plausible_answers = qas['plausible_answers']
        for plausible_answer in plausible_answers:
          plausible_answer['answer_start'] = find_continuous_span_start(modified_contexts_keep[context_index], plausible_answer['text'])

with open('./dev-v2.0-np-pre.json', 'w') as f:
  json.dump(d, f)

d1 = Load_json_data('./dev-v2.0-np-pre.json')

titles, np_contexts, np_questions, questions_preserved = [], [], [], []

for title_id in tqdm.tqdm(range(len(d1['data']))):
  titles.append(d1['data'][title_id]['title'])
  contexts_for_each_title, questions = [], []
  paragraphs = d1['data'][title_id]['paragraphs']
  for paragraph in paragraphs:
    contexts_for_each_title.append(paragraph['context'])
    questions_for_each_context = []
    qass = paragraph['qas']
    for qas in qass:
      answer_starts = []
      if str(qas['is_impossible']) == "False":
        answers = qas['answers']
        for answer in answers:
          answer_starts.append(answer['answer_start'])
        if -1 not in answer_starts:
          questions_for_each_context.append(qas['question'])
      else:
        plausible_answers = qas['plausible_answers']
        for plausible_answer in plausible_answers:
          answer_starts.append(plausible_answer['answer_start'])
        if -1 not in answer_starts:
          questions_for_each_context.append(qas['question'])
    questions_preserved.append(questions_for_each_context)
    questions.append(questions_for_each_context)
  np_questions.append(questions)
  np_contexts.append(contexts_for_each_title)

d2 = Load_json_data('./dev-v2.0-original-pre.json')

ori_contexts = []

for title_id in tqdm.tqdm(range(len(d2['data']))):
  contexts_for_each_title = []
  paragraphs = d2['data'][title_id]['paragraphs']
  for paragraph in paragraphs:
    contexts_for_each_title.append(paragraph['context'])
  ori_contexts.append(contexts_for_each_title)

def sublist_lengths(lst):
    return [len(sublist) for sublist in lst]

def flatten_list(lst):
    return [item for sublist in lst for item in sublist]

k = [sublist_lengths(np_questions[id]) for id in range(len(np_questions))]
titles_p = [titles[id] for id in range(len(titles)) if list(set(k[id])) != [0]]
np_contexts_delete = [flatten_list(np_contexts)[id] for id in range(len(flatten_list(np_contexts))) if flatten_list(k)[id] == 0]
ori_contexts_delete = [flatten_list(ori_contexts)[id] for id in range(len(flatten_list(ori_contexts))) if flatten_list(k)[id] == 0]

d4 = Load_json_data('./dev-v2.0-original-pre.json')

for title_id in tqdm.tqdm(range(len(d4['data']))):
  d4['data'] = [item for item in d4['data'] if item['title'] in titles_p]

for title_id in tqdm.tqdm(range(len(d4['data']))):
  paragraphs = d4['data'][title_id]['paragraphs']
  d4['data'][title_id]['paragraphs'] = [item for item in paragraphs if item['context'] not in ori_contexts_delete]
  paragraphs_new = [item for item in paragraphs if item['context'] not in ori_contexts_delete]
  for paragraph in paragraphs_new:
    context = paragraph['context']
    context_index = flatten_list(ori_contexts).index(context)
    qass = paragraph['qas']
    paragraph['qas'] = [item for item in qass if item['question'] in questions_preserved[context_index]]

with open('./dev-v2.0-original-new.json', 'w') as f:
  json.dump(d4, f)

d5 = Load_json_data('./dev-v2.0-original-new.json')

for title_id in tqdm.tqdm(range(len(d5['data']))):
  paragraphs = d5['data'][title_id]['paragraphs']
  for paragraph in paragraphs:
    context = paragraph['context']
    context_index = flatten_list(ori_contexts).index(context)
    paragraph['context'] = flatten_list(np_contexts)[context_index]
    qass = paragraph['qas']
    for qas in qass:
      if str(qas['is_impossible']) == "False":
        answers = qas['answers']
        for answer in answers:
          gta = answer['text']
          answer['answer_start'] = find_continuous_span_start(flatten_list(np_contexts)[context_index], gta)
      else:
        plausible_answers = qas['plausible_answers']
        for plausible_answer in plausible_answers:
          plausible_answer['answer_start'] = find_continuous_span_start(flatten_list(np_contexts)[context_index], plausible_answer['text'])

with open('./dev-v2.0-np-new.json', 'w') as f:
  json.dump(d5, f)