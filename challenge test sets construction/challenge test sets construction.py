import os
import pandas as pd
import json

def list_eval_predictions_paths(folder_path):
  subfolders = [f.path+'/eval_predictions.json' for f in os.scandir(folder_path) if f.is_dir()]
  return subfolders

def Load_json_data(file_path):
  file = open(file_path, 'r')
  data = json.load(file)
  file.close()
  return data

def Load_Read_SQuAD_Json_File_f(file_path):
  json_file = open(file_path, 'r')
  squad_data = json.load(json_file)
  json_file.close()

  titles, contexts, questions, question_ids, ground_truth_answers, ground_truth_answer_starts = [], [], [], [], [], []

  for article in squad_data['data']:
    titles.append(article['title'])

    title_contexts, title_questions, title_qids, title_answers, title_answer_starts = [], [], [], [], []

    for paragraph in article['paragraphs']:
      title_contexts.append(paragraph['context'])

      context_questions, context_qids, context_answers, context_answer_starts = [], [], [], []

      for qa in paragraph['qas']:
        context_questions.append(qa['question'])
        context_qids.append(qa['id'])
        context_answers.append([answer['text'] for answer in qa['answers']])
        context_answer_starts.append([answer['answer_start'] for answer in qa['answers']])

      title_questions.append(context_questions)
      title_qids.append(context_qids)
      title_answers.append(context_answers)
      title_answer_starts.append(context_answer_starts)

    contexts.append(title_contexts)
    questions.append(title_questions)
    question_ids.append(title_qids)
    ground_truth_answers.append(title_answers)
    ground_truth_answer_starts.append(title_answer_starts)

  return [titles, contexts, questions, question_ids, ground_truth_answers, ground_truth_answer_starts]

def Load_Read_SQuAD_Json_File_s(file_path):
  json_file = open(file_path, 'r')
  squad_data = json.load(json_file)
  json_file.close()

  titles, contexts, questions, question_ids, ground_truth_answers, ground_truth_answer_starts = [], [], [], [], [], []

  for article in squad_data['data']:
    titles.append(article['title'])

    title_contexts, title_questions, title_qids, title_answers, title_answer_starts = [], [], [], [], []

    for paragraph in article['paragraphs']:
      title_contexts.append(paragraph['context'])

      context_questions, context_qids, context_answers, context_answer_starts = [], [], [], []

      for qa in paragraph['qas']:
        if str(qa['is_impossible']) == "false":
          context_questions.append(qa['question'])
          context_qids.append(qa['id'])
          context_answers.append([answer['text'] for answer in qa['answers']])
          context_answer_starts.append([answer['answer_start'] for answer in qa['answers']])
        else:
          context_questions.append(qa['question'])
          context_qids.append(qa['id'])
          context_answers.append([answer['text'] for answer in qa['plausible_answers']])
          context_answer_starts.append([answer['answer_start'] for answer in qa['plausible_answers']])

      title_questions.append(context_questions)
      title_qids.append(context_qids)
      title_answers.append(context_answers)
      title_answer_starts.append(context_answer_starts)

    contexts.append(title_contexts)
    questions.append(title_questions)
    question_ids.append(title_qids)
    ground_truth_answers.append(title_answers)
    ground_truth_answer_starts.append(title_answer_starts)

  return [titles, contexts, questions, question_ids, ground_truth_answers, ground_truth_answer_starts]

import re

import string

import collections

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

model_name_or_paths0 = [
    'distilbert-base-fv',
    'bert-base-cased-fv',
    'bert-base-uncased-fv',
    'bert-large-cased-fv',
    'bert-large-uncased-fv',
    'spanbert-base-cased-fv',
    'spanbert-large-cased-fv',
    'roberta-base-fv',
    'roberta-large-fv',
    'albert-base-v1-fv',
    'albert-base-v2-fv',
    'albert-large-v1-fv',
    'albert-large-v2-fv',
    'albert-xxlarge-v1-fv',
    'albert-xxlarge-v2-fv',
    'deberta-large-fv'
]

model_name_or_paths1 = [
    'distilbert-base-sv',
    'bert-base-cased-sv',
    'bert-base-uncased-sv',
    'bert-large-cased-sv',
    'bert-large-uncased-sv',
    'spanbert-base-cased-sv',
    'spanbert-large-cased-sv',
    'roberta-base-sv',
    'roberta-large-sv',
    'albert-base-v1-sv',
    'albert-base-v2-sv',
    'albert-large-v1-sv',
    'albert-large-v2-sv',
    'albert-xxlarge-v1-sv',
    'albert-xxlarge-v2-sv',
    'deberta-large-sv'
]

def find_substring_position(string, substring):
    return string.find(substring)

def Find_Element_All_Occurrences(example_list, element):
  return [index for index in range(len(example_list)) if example_list[index] == element]

import random

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

all_files = list_eval_predictions_paths('./')

datas = ['fvcomprehensiveasc', 'svcomprehensiveasc']

def method(data_id):
  if data_id == 0:
    fv_files = [f for f in all_files if datas[data_id] in f]
    print(len(fv_files))
    dicts = []
    model_name_or_paths = globals()[f"model_name_or_paths{data_id}"]
    for model_name_or_path in model_name_or_paths:
      files = [f for f in fv_files if model_name_or_path in f and find_substring_position(f, model_name_or_path)==2]
      dicts.append(Load_json_data([f for f in files if '-ori' in f][0]))
      dicts.append(Load_json_data([f for f in files if '-mod' in f][0]))

    ori = Load_Read_SQuAD_Json_File_f('../comprehensive-fvsv-asc/fvoriginal.json')
    mod = Load_Read_SQuAD_Json_File_f('../comprehensive-fvsv-asc/fvmodified.json')

    titles = ori[0]
    contexts = ori[1]
    mod_contexts = mod[1]
    questions = ori[2]
    question_ids = ori[3]
    ground_truth_answers = ori[4]
    ground_truth_answer_starts = ori[5]
    mod_ground_truth_answer_starts = mod[5]

    ntitles, ncontexts, nmodi_contexts, nquestions, nquestion_ids, nground_truth_answers, nground_truth_answer_starts_o, nground_truth_answer_starts_m = [], [], [], [], [], [], [], []

    for title_id in range(len(titles)):
      title_contexts, title_modi_contexts, title_questions, title_qids, title_answers, title_answer_starts_o, title_answer_starts_m = [], [], [], [], [], [], []
      contexts_for_this_title = contexts[title_id]
      mod_contexts_for_this_title = mod_contexts[title_id]
      for context_id in range(len(contexts_for_this_title)):
        context_questions, context_qids, context_answers, context_answer_starts_o, context_answer_starts_m = [], [], [], [], []

        context = contexts_for_this_title[context_id]
        mod_context = mod_contexts_for_this_title[context_id]
        qs = questions[title_id][context_id]
        qids = question_ids[title_id][context_id]
        gtas = ground_truth_answers[title_id][context_id]
        gta_starts = ground_truth_answer_starts[title_id][context_id]
        mod_gta_starts = mod_ground_truth_answer_starts[title_id][context_id]

        asc_statistics, desc_statistics = [], []

        for question_id in range(len(qs)):
          qid = qids[question_id]
          prediction_all_models = [dic[qid] for dic in dicts]
          asc_statistics_temp, desc_statistics_temp = [], []
          indices = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
          for index in indices:
            if max(compute_exact(a_gold, prediction_all_models[index]) for a_gold in gtas[question_id])==1 and max(compute_f1(a_gold, prediction_all_models[index+1]) for a_gold in gtas[question_id])<0.4:
              asc_statistics_temp.append(index)
            if max(compute_f1(a_gold, prediction_all_models[index]) for a_gold in gtas[question_id])<0.4 and max(compute_exact(a_gold, prediction_all_models[index+1]) for a_gold in gtas[question_id])==1:
              desc_statistics_temp.append(index)
          asc_statistics.append(len(asc_statistics_temp)/len(indices))
          desc_statistics.append(len(desc_statistics_temp)/len(indices))

        asc_statistics_result = sum(asc_statistics)
        desc_statistics_result = sum(desc_statistics)

        if asc_statistics_result!=0 and asc_statistics_result>desc_statistics_result:
          zero_indices = Find_Element_All_Occurrences(asc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(context)
          title_modi_contexts.append(mod_context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
        elif desc_statistics_result!=0 and desc_statistics_result>asc_statistics_result:
          zero_indices = Find_Element_All_Occurrences(desc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(mod_context)
          title_modi_contexts.append(context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
        elif asc_statistics_result==desc_statistics_result and asc_statistics_result!=0 and desc_statistics_result!=0:
          zero_indices = Find_Element_All_Occurrences(asc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(context)
          title_modi_contexts.append(mod_context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
      if len(title_contexts)>0:
        ntitles.append(titles[title_id])
        ncontexts.append(title_contexts)
        nmodi_contexts.append(title_modi_contexts)
        nquestions.append(title_questions)
        nquestion_ids.append(title_qids)
        nground_truth_answers.append(title_answers)
        nground_truth_answer_starts_o.append(title_answer_starts_o)
        nground_truth_answer_starts_m.append(title_answer_starts_m)

    ftitles, fcontexts, fmodi_contexts, fquestions, fquestion_ids, fground_truth_answers, fground_truth_answer_starts_o, fground_truth_answer_starts_m = [], [], [], [], [], [], [], []

    for id in range(len(ntitles)):
      dindices = [ind for ind in range(len(ncontexts[id])) if ncontexts[id][ind] not in [item for sublist in contexts for item in sublist]]
      ncontextsnew = [ncontexts[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
      if len(ncontextsnew)>0:
        nmodi_contextsnew = [nmodi_contexts[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nquestionsnew = [nquestions[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nquestion_idsnew = [nquestion_ids[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answersnew = [nground_truth_answers[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answer_starts_onew = [nground_truth_answer_starts_o[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answer_starts_mnew = [nground_truth_answer_starts_m[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        ori_two_times = list(set([item for item in ncontextsnew if ncontextsnew.count(item) > 1]))
        if len(ori_two_times)>0:
          delete_indices = []
          for item in ori_two_times:
            tt_indices = Find_Element_All_Occurrences(ncontextsnew, item)
            qs_lengths = [len(nquestionsnew[index]) for index in tt_indices]
            max_length_index = qs_lengths.index(max(qs_lengths))
            for j in range(len(tt_indices)):
              if j != max_length_index:
                delete_indices.append(tt_indices[j])
          ftitles.append(ntitles[id])
          fcontexts.append([ncontextsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fmodi_contexts.append([nmodi_contextsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fquestions.append([nquestionsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fquestion_ids.append([nquestion_idsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answers.append([nground_truth_answersnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answer_starts_o.append([nground_truth_answer_starts_onew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answer_starts_m.append([nground_truth_answer_starts_mnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])

        else:
          ftitles.append(ntitles[id])
          fcontexts.append(ncontextsnew)
          fmodi_contexts.append(nmodi_contextsnew)
          fquestions.append(nquestionsnew)
          fquestion_ids.append(nquestion_idsnew)
          fground_truth_answers.append(nground_truth_answersnew)
          fground_truth_answer_starts_o.append(nground_truth_answer_starts_onew)
          fground_truth_answer_starts_m.append(nground_truth_answer_starts_mnew)

    return ftitles, fcontexts, fmodi_contexts, fquestions, fquestion_ids, fground_truth_answers, fground_truth_answer_starts_o, fground_truth_answer_starts_m

  elif data_id==1:
    fv_files = [f for f in all_files if datas[data_id] in f]
    print(len(fv_files))
    dicts = []
    model_name_or_paths = globals()[f"model_name_or_paths{data_id}"]
    for model_name_or_path in model_name_or_paths:
      files = [f for f in fv_files if model_name_or_path in f and find_substring_position(f, model_name_or_path)==2]
      dicts.append(Load_json_data([f for f in files if '-ori' in f][0]))
      dicts.append(Load_json_data([f for f in files if '-mod' in f][0]))

    ori = Load_Read_SQuAD_Json_File_s('../comprehensive-fvsv-asc/svoriginal.json')
    mod = Load_Read_SQuAD_Json_File_s('../comprehensive-fvsv-asc/svmodified.json')

    titles = ori[0]
    contexts = ori[1]
    mod_contexts = mod[1]
    questions = ori[2]
    question_ids = ori[3]
    ground_truth_answers = ori[4]
    ground_truth_answer_starts = ori[5]
    mod_ground_truth_answer_starts = mod[5]

    ntitles, ncontexts, nmodi_contexts, nquestions, nquestion_ids, nground_truth_answers, nground_truth_answer_starts_o, nground_truth_answer_starts_m = [], [], [], [], [], [], [], []

    for title_id in range(len(titles)):
      title_contexts, title_modi_contexts, title_questions, title_qids, title_answers, title_answer_starts_o, title_answer_starts_m = [], [], [], [], [], [], []
      contexts_for_this_title = contexts[title_id]
      mod_contexts_for_this_title = mod_contexts[title_id]
      for context_id in range(len(contexts_for_this_title)):
        context_questions, context_qids, context_answers, context_answer_starts_o, context_answer_starts_m = [], [], [], [], []

        context = contexts_for_this_title[context_id]
        mod_context = mod_contexts_for_this_title[context_id]
        qs = questions[title_id][context_id]
        qids = question_ids[title_id][context_id]
        gtas = ground_truth_answers[title_id][context_id]
        gta_starts = ground_truth_answer_starts[title_id][context_id]
        mod_gta_starts = mod_ground_truth_answer_starts[title_id][context_id]

        asc_statistics, desc_statistics = [], []

        for question_id in range(len(qs)):
          qid = qids[question_id]
          prediction_all_models = [dic[qid] for dic in dicts]
          if len(gtas[question_id]) == 1:
            asc_statistics_temp, desc_statistics_temp = [], []
            indices = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
            for index in indices:
              if len(prediction_all_models[index])==0 and len(prediction_all_models[index+1])>0:
                asc_statistics_temp.append(index)
              if len(prediction_all_models[index])>0 and len(prediction_all_models[index+1])==0:
                desc_statistics_temp.append(index)
            asc_statistics.append(len(asc_statistics_temp)/len(indices))
            desc_statistics.append(len(desc_statistics_temp)/len(indices))
          elif len(gtas[question_id])>1:
            asc_statistics_temp, desc_statistics_temp = [], []
            indices = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
            for index in indices:
              if len(prediction_all_models[index])>0 and len(prediction_all_models[index+1])>0:
                if max(compute_exact(a_gold, prediction_all_models[index]) for a_gold in gtas[question_id])==1 and max(compute_f1(a_gold, prediction_all_models[index+1]) for a_gold in gtas[question_id])<0.4:
                  asc_statistics_temp.append(index)
                if max(compute_f1(a_gold, prediction_all_models[index]) for a_gold in gtas[question_id])<0.4 and max(compute_exact(a_gold, prediction_all_models[index+1]) for a_gold in gtas[question_id])==1:
                  desc_statistics_temp.append(index)
              if len(prediction_all_models[index])>0 and len(prediction_all_models[index+1])==0:
                if max(compute_exact(a_gold, prediction_all_models[index]) for a_gold in gtas[question_id])==1:
                  asc_statistics_temp.append(index)
              if len(prediction_all_models[index])==0 and len(prediction_all_models[index+1])>0:
                if max(compute_exact(a_gold, prediction_all_models[index+1]) for a_gold in gtas[question_id])==1:
                  desc_statistics_temp.append(index)
            asc_statistics.append(len(asc_statistics_temp)/len(indices))
            desc_statistics.append(len(desc_statistics_temp)/len(indices))

        asc_statistics_result = sum(asc_statistics)
        desc_statistics_result = sum(desc_statistics)

        if asc_statistics_result!=0 and asc_statistics_result>desc_statistics_result:
          zero_indices = Find_Element_All_Occurrences(asc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(context)
          title_modi_contexts.append(mod_context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
        elif desc_statistics_result!=0 and desc_statistics_result>asc_statistics_result:
          zero_indices = Find_Element_All_Occurrences(desc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(mod_context)
          title_modi_contexts.append(context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
        elif asc_statistics_result==desc_statistics_result and asc_statistics_result!=0 and desc_statistics_result!=0:
          zero_indices = Find_Element_All_Occurrences(asc_statistics, 0)

          context_questions = [qs[id] for id in range(len(qs)) if id not in zero_indices]
          context_qids = [qids[id] for id in range(len(qs)) if id not in zero_indices]
          context_answers = [gtas[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_o = [gta_starts[id] for id in range(len(qs)) if id not in zero_indices]
          context_answer_starts_m = [mod_gta_starts[id] for id in range(len(qs)) if id not in zero_indices]

          title_contexts.append(context)
          title_modi_contexts.append(mod_context)
          title_questions.append(context_questions)
          title_qids.append(context_qids)
          title_answers.append(context_answers)
          title_answer_starts_o.append(context_answer_starts_o)
          title_answer_starts_m.append(context_answer_starts_m)
      if len(title_contexts)>0:
        ntitles.append(titles[title_id])
        ncontexts.append(title_contexts)
        nmodi_contexts.append(title_modi_contexts)
        nquestions.append(title_questions)
        nquestion_ids.append(title_qids)
        nground_truth_answers.append(title_answers)
        nground_truth_answer_starts_o.append(title_answer_starts_o)
        nground_truth_answer_starts_m.append(title_answer_starts_m)

    ftitles, fcontexts, fmodi_contexts, fquestions, fquestion_ids, fground_truth_answers, fground_truth_answer_starts_o, fground_truth_answer_starts_m = [], [], [], [], [], [], [], []

    for id in range(len(ntitles)):
      dindices = [ind for ind in range(len(ncontexts[id])) if ncontexts[id][ind] not in [item for sublist in contexts for item in sublist]]
      ncontextsnew = [ncontexts[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
      if len(ncontextsnew)>0:
        nmodi_contextsnew = [nmodi_contexts[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nquestionsnew = [nquestions[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nquestion_idsnew = [nquestion_ids[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answersnew = [nground_truth_answers[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answer_starts_onew = [nground_truth_answer_starts_o[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        nground_truth_answer_starts_mnew = [nground_truth_answer_starts_m[id][i] for i in range(len(ncontexts[id])) if i not in dindices]
        ori_two_times = list(set([item for item in ncontextsnew if ncontextsnew.count(item) > 1]))
        if len(ori_two_times)>0:
          delete_indices = []
          for item in ori_two_times:
            tt_indices = Find_Element_All_Occurrences(ncontextsnew, item)
            qs_lengths = [len(nquestionsnew[index]) for index in tt_indices]
            max_length_index = qs_lengths.index(max(qs_lengths))
            for j in range(len(tt_indices)):
              if j != max_length_index:
                delete_indices.append(tt_indices[j])
          ftitles.append(ntitles[id])
          fcontexts.append([ncontextsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fmodi_contexts.append([nmodi_contextsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fquestions.append([nquestionsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fquestion_ids.append([nquestion_idsnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answers.append([nground_truth_answersnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answer_starts_o.append([nground_truth_answer_starts_onew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])
          fground_truth_answer_starts_m.append([nground_truth_answer_starts_mnew[k] for k in range(len(ncontextsnew)) if k not in delete_indices])

        else:
          ftitles.append(ntitles[id])
          fcontexts.append(ncontextsnew)
          fmodi_contexts.append(nmodi_contextsnew)
          fquestions.append(nquestionsnew)
          fquestion_ids.append(nquestion_idsnew)
          fground_truth_answers.append(nground_truth_answersnew)
          fground_truth_answer_starts_o.append(nground_truth_answer_starts_onew)
          fground_truth_answer_starts_m.append(nground_truth_answer_starts_mnew)

    return ftitles, fcontexts, fmodi_contexts, fquestions, fquestion_ids, fground_truth_answers, fground_truth_answer_starts_o, fground_truth_answer_starts_m

fvtitles, fvcontexts, fvmodi_contexts, fvquestions, fvquestion_ids, fvground_truth_answers, fvground_truth_answer_starts_o, fvground_truth_answer_starts_m = method(0)

import json

data_list = []

for article_id in range(len(fvtitles)):
  paragraphs_list = []
  for passage_id in range(len(fvcontexts[article_id])):
    qas_list = []
    for question_id in range(len(fvquestions[article_id][passage_id])):
      answers_list = []
      for answer_id in range(len(fvground_truth_answers[article_id][passage_id][question_id])):
        answers_list.append({"answer_start": fvground_truth_answer_starts_o[article_id][passage_id][question_id][answer_id], "text": fvground_truth_answers[article_id][passage_id][question_id][answer_id]})
      qas_list.append({"answers": answers_list, "question": fvquestions[article_id][passage_id][question_id], "id": fvquestion_ids[article_id][passage_id][question_id]})
    paragraphs_list.append({"context": fvcontexts[article_id][passage_id], "qas": qas_list})
  data_list.append({"title": fvtitles[article_id], "paragraphs": paragraphs_list})

my_dict_o = {"data": data_list, "version": "1.1"}

file_path = './fvoriginal-new.json'

with open(file_path, 'w') as json_file:
  json.dump(my_dict_o, json_file)

import json

data_list = []

for article_id in range(len(fvtitles)):
  paragraphs_list = []
  for passage_id in range(len(fvmodi_contexts[article_id])):
    qas_list = []
    for question_id in range(len(fvquestions[article_id][passage_id])):
      answers_list = []
      for answer_id in range(len(fvground_truth_answers[article_id][passage_id][question_id])):
        answers_list.append({"answer_start": fvground_truth_answer_starts_m[article_id][passage_id][question_id][answer_id], "text": fvground_truth_answers[article_id][passage_id][question_id][answer_id]})
      qas_list.append({"answers": answers_list, "question": fvquestions[article_id][passage_id][question_id], "id": fvquestion_ids[article_id][passage_id][question_id]})
    paragraphs_list.append({"context": fvmodi_contexts[article_id][passage_id], "qas": qas_list})
  data_list.append({"title": fvtitles[article_id], "paragraphs": paragraphs_list})

my_dict_m = {"data": data_list, "version": "1.1"}

file_path = './fvmodified-new.json'

with open(file_path, 'w') as json_file:
  json.dump(my_dict_m, json_file)


svtitles, svcontexts, svmodi_contexts, svquestions, svquestion_ids, svground_truth_answers, svground_truth_answer_starts_o, svground_truth_answer_starts_m = method(1)

import json

data_list = []

for article_id in range(len(svtitles)):
  paragraphs_list = []
  for passage_id in range(len(svcontexts[article_id])):
    qas_list = []
    for question_id in range(len(svquestions[article_id][passage_id])):
      answers_list = []

      for answer_id in range(len(svground_truth_answers[article_id][passage_id][question_id])):
        answers_list.append({"text": svground_truth_answers[article_id][passage_id][question_id][answer_id], "answer_start": svground_truth_answer_starts_o[article_id][passage_id][question_id][answer_id]})
      if len(svground_truth_answers[article_id][passage_id][question_id]) > 1:
        qas_list.append({"question": svquestions[article_id][passage_id][question_id], "id": svquestion_ids[article_id][passage_id][question_id], "answers": answers_list, "is_impossible": 'false'})
      else:
        qas_list.append({"plausible_answers": answers_list, "question": svquestions[article_id][passage_id][question_id], "id": svquestion_ids[article_id][passage_id][question_id], "answers": [], "is_impossible": 'true'})
    paragraphs_list.append({"qas": qas_list, "context": svcontexts[article_id][passage_id]})
  data_list.append({"title": svtitles[article_id], "paragraphs": paragraphs_list})

my_dict_o = {"version": "v2.0", "data": data_list}

file_path = './svoriginal-new.json'

with open(file_path, 'w') as json_file:
  json.dump(my_dict_o, json_file)

import json

data_list = []

for article_id in range(len(svtitles)):
  paragraphs_list = []
  for passage_id in range(len(svmodi_contexts[article_id])):
    qas_list = []
    for question_id in range(len(svquestions[article_id][passage_id])):
      answers_list = []

      for answer_id in range(len(svground_truth_answers[article_id][passage_id][question_id])):
        answers_list.append({"text": svground_truth_answers[article_id][passage_id][question_id][answer_id], "answer_start": svground_truth_answer_starts_m[article_id][passage_id][question_id][answer_id]})
      if len(svground_truth_answers[article_id][passage_id][question_id]) > 1:
        qas_list.append({"question": svquestions[article_id][passage_id][question_id], "id": svquestion_ids[article_id][passage_id][question_id], "answers": answers_list, "is_impossible": 'false'})
      else:
        qas_list.append({"plausible_answers": answers_list, "question": svquestions[article_id][passage_id][question_id], "id": svquestion_ids[article_id][passage_id][question_id], "answers": [], "is_impossible": 'true'})
    paragraphs_list.append({"qas": qas_list, "context": svmodi_contexts[article_id][passage_id]})
  data_list.append({"title": svtitles[article_id], "paragraphs": paragraphs_list})

my_dict_m = {"version": "v2.0", "data": data_list}

file_path = './svmodified-new.json'

with open(file_path, 'w') as json_file:
  json.dump(my_dict_m, json_file)
