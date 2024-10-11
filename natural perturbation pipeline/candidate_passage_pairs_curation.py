#!/usr/bin/env python

import subprocess

subprocess.run(["pip", "install", "wikipedia"])
subprocess.run(["pip", "install", "wikipedia-api"])
subprocess.run(["pip", "install", "wikitextparser"])

import wikipedia
import re
import numpy as np
from collections import Counter
from datetime import datetime

import wikitextparser
import requests

def get_all_versions_data_content_tags_removed(page_title):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content|timestamp|user|size|comment|tags",
        "format": "json",
        "rvlimit": 500,  # Set a high limit for revisions
    }

    all_versions_data = []

    try:

      while True:
        response = requests.get(base_url, params=params)
        data = response.json()

        page_id = list(data["query"]["pages"].keys())[0]
        revisions = data["query"]["pages"][page_id]["revisions"]

        for revision in revisions:
          content_wikitext = revision.get("*", "Content not available")
          content_text = wikitextparser.parse(content_wikitext).plain_text()

          timestamp = revision.get("timestamp", "Timestamp not available")
          user = revision.get("user", "User not available")
          size = revision.get("size", "Size not available")
          comment = revision.get("comment", "Comment not available")
          tags = revision.get("tags", "Tags not available")

          revision_data = {
                  "content": content_text,
                  "timestamp": timestamp,
                  "user": user,
                  "size": size,
                  "comment": comment,
                  "tags": tags,
          }

          all_versions_data.append(revision_data)

        if "continue" in data:
          params["rvcontinue"] = data["continue"]["rvcontinue"]
        else:
          break

    except KeyError as e:
      return f"Error: {e}. 'revisions' key not found in the response data."

    return all_versions_data

def get_all_versions_data(page_title):
    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content|timestamp|user|size|comment|tags",
        "format": "json",
        "rvlimit": 500,  # Set a high limit for revisions
    }

    all_versions_data = []

    try:

      while True:
          response = requests.get(base_url, params=params)
          data = response.json()

          page_id = list(data["query"]["pages"].keys())[0]
          revisions = data["query"]["pages"][page_id]["revisions"]

          for revision in revisions:
              content = revision.get("*", "Content not available")
              timestamp = revision.get("timestamp", "Timestamp not available")
              user = revision.get("user", "User not available")
              size = revision.get("size", "Size not available")
              comment = revision.get("comment", "Comment not available")
              tags = revision.get("tags", "Tags not available")

              revision_data = {
                  "content": content,
                  "timestamp": timestamp,
                  "user": user,
                  "size": size,
                  "comment": comment,
                  "tags": tags,
              }

              all_versions_data.append(revision_data)

          if "continue" in data:
              params["rvcontinue"] = data["continue"]["rvcontinue"]
          else:
              break

    except KeyError as e:
      return f"Error: {e}. 'revisions' key not found in the response data."

    return all_versions_data

def group_items_with_distance_one(lst):
    result = []
    current_group = [lst[0]]

    for i in range(1, len(lst)):
        if abs(lst[i] - lst[i - 1]) == 1:
            current_group.append(lst[i])
        else:
            result.append(current_group)
            current_group = [lst[i]]

    result.append(current_group)

    return result

def compare_strings(str1, str2):
  min_len = min(len(str1), len(str2))
  replace_changes, delete_changes, add_changes = [], [], []
  for i in range(min_len):
    if str1[i] != str2[i]:
      replace_changes.append({'position': i, 'before': str1[i], 'after': str2[i]})
  if len(str1) > min_len:
    for i in range(min_len, len(str1)):
      delete_changes.append({'position': i, 'deleted_character': str1[i]})
  elif len(str2) > min_len:
    for i in range(min_len, len(str2)):
      add_changes.append({'position': i, 'added_character': str2[i]})

  records_replace, records_delete, records_add = [], [], []

  if len(replace_changes) != 0:
    result_replace_changes = group_items_with_distance_one([replace_change['position'] for replace_change in replace_changes])
    for index in range(len(result_replace_changes)):
      if len(result_replace_changes[index]) == 1:
        corresponding_replace_change = [rchange for rchange in replace_changes if rchange['position']==result_replace_changes[index][0]]
        record = f'At position {corresponding_replace_change[0]["position"]}, replace "{corresponding_replace_change[0]["before"]}" with "{corresponding_replace_change[0]["after"]}"'
        records_replace.append(record)
      else:
        corresponding_replace_changes = [rchange for rchange in replace_changes if rchange['position'] in result_replace_changes[index]]
        corresponding_before = [crchange['before'] for crchange in corresponding_replace_changes]
        corresponding_after = [crchange['after'] for crchange in corresponding_replace_changes]
        record = f'At position {", ".join([str(item) for item in result_replace_changes[index]])}, replace "{"".join(corresponding_before)}" with "{"".join(corresponding_after)}"'
        records_replace.append(record)
  if len(delete_changes) != 0:
    result_delete_changes = group_items_with_distance_one([delete_change['position'] for delete_change in delete_changes])
    for index in range(len(result_delete_changes)):
      if len(result_delete_changes[index]) == 1:
        corresponding_delete_change = [dchange for dchange in delete_changes if dchange['position']==result_delete_changes[index][0]]
        record = f'At position {corresponding_delete_change[0]["position"]}, delete "{corresponding_delete_change[0]["deleted_character"]}"'
        records_delete.append(record)
      else:
        corresponding_delete_changes = [dchange for dchange in delete_changes if dchange['position'] in result_delete_changes[index]]
        corresponding_delete = [cdchange['deleted_character'] for cdchange in corresponding_delete_changes]
        record = f'At position {", ".join([str(item) for item in result_delete_changes[index]])}, delete "{"".join(corresponding_delete)}"'
        records_delete.append(record)
  if len(add_changes) != 0:
    result_add_changes = group_items_with_distance_one([add_change['position'] for add_change in add_changes])
    for index in range(len(result_add_changes)):
      if len(result_add_changes[index]) == 1:
        corresponding_add_change = [achange for achange in add_changes if achange['position']==result_add_changes[index][0]]
        record = f'At position {corresponding_add_change[0]["position"]}, add "{corresponding_add_change[0]["added_character"]}"'
        records_add.append(record)
      else:
        corresponding_add_changes = [achange for achange in add_changes if achange['position'] in result_add_changes[index]]
        corresponding_add = [cachange['added_character'] for cachange in corresponding_add_changes]
        record = f'At position {", ".join([str(item) for item in result_add_changes[index]])}, add "{"".join(corresponding_add)}"'
        records_add.append(record)
  return records_replace+records_delete+records_add

import difflib

differ = difflib.Differ()

def vandalism_pattern_analysis(list1, list2):
  diff = list(differ.compare(list1, list2))
  changes = []
  valid_changes = []
  position_list1 = 0
  position_list2 = 0
  full_revision_details = []
  original_content, modified_content = [], []
  for i, line in enumerate(diff):
    if line.startswith('  '):
      changes.append({"type": "unchanged", "value": line[2:], "position": position_list1})
      position_list1 += 1
      position_list2 += 1
    elif line.startswith('- '):
      changes.append({"type": "removed", "value": line[2:], "position": position_list1})
      position_list1 += 1
    elif line.startswith('+ '):
      changes.append({"type": "added", "value": line[2:], "position": position_list2})
      position_list2 += 1
  for change in changes:
    if change['type'] == 'unchanged':
      message_unchanged = f"At position {change['position']}, item '''{change['value']}''' remains {change['type']}"
      full_revision_details.append(message_unchanged)
      # print(message_unchanged)
    else:
      valid_changes.append(change)
  position_value_list = list(set([valid_change['position'] for valid_change in valid_changes]))
  for position_value in position_value_list:
    length_test = [valid_change for valid_change in valid_changes if valid_change['position']==position_value]
    if len(length_test)==2 and len(list(set([item['type'] for item in length_test])))==2:
      for v_change in length_test:
        if v_change['type'] == 'removed':
          original_value = v_change['value']
          original_content.append(original_value)
        if v_change['type'] == 'added':
          modified_value = v_change['value']
          modified_content.append(modified_value)
      message_changed = f"At position {position_value}, item '''{original_value}''' changed to '''{modified_value}'''"
      # modification_details = compare_strings(original_value, modified_value)
      # full_revision_details.append(message_changed+"\n"+"Details:"+"\n"+"\n".join(modification_details))
      full_revision_details.append(message_changed)
    if len(length_test)==1:
      if length_test[0]['type'] == 'removed':
        original_value = length_test[0]['value']
        original_content.append(original_value)
        modified_value = ''
        modified_content.append(modified_value)
        message_changed = f"At position {position_value}, item '''{original_value}''' has been deleted"
        full_revision_details.append(message_changed)
      if length_test[0]['type'] == 'added':
        modified_value = length_test[0]['value']
        modified_content.append(modified_value)
        original_value = ''
        original_content.append(original_value)
        message_changed = f"At position {position_value}, item '''{modified_value}''' has been added"
        full_revision_details.append(message_changed)
  return [full_revision_details, original_content, modified_content]

def generate_sublists(original_list):
  sublists = []

  for i in range(10):
    lower_bound = i * 10
    upper_bound = (i + 1) * 10
    sublist = [item for item in original_list if lower_bound <= item < upper_bound]
    sublists.append(sublist)

  return sublists

def concatenate_with_none(a, b):
    result = []

    for i, item in enumerate(a):
        result.append(item)

        if len(b[i]) > 1:
            result.extend([''] * (len(b[i]) - 1))

    return result

def flatten_list_with_nan(b):
    result = []

    for sublist in b:
        if not sublist:  # Check if sublist is empty
            result.append('')
        else:
            result.extend(sublist)

    return result

import pandas as pd

def save_to_csv(mcp, ocp, mc, oc, t, u, s, c, time, inf, inf_p, potential_original_reading_passages, potential_modified_reading_passages_vandalism, csv_filename):
  data = {'title': [], 'time': [], 'user': [], 'size': [], 'comment': [], 'tags': [], 'original passages': [], 'modified_passages': []}

  for key, value in potential_original_reading_passages.items():
    data['title'].append(key)
    c1 = concatenate_with_none(oc[key], value)

    c2 = concatenate_with_none(mc[key], value)

    c3 = concatenate_with_none(inf[key], value)

    c4 = concatenate_with_none(time[key], value)

    c5 = concatenate_with_none(u[key], value)

    c6 = concatenate_with_none(s[key], value)

    c7 = concatenate_with_none(c[key], value)

    c8 = concatenate_with_none(t[key], value)

    c9 = concatenate_with_none(ocp[key], value)

    c10 = concatenate_with_none(mcp[key], value)

    c11 = concatenate_with_none(inf_p[key], value)

    c12 = [str(item) for item in flatten_list_with_nan(value)]
    c13 = [str(item) for item in flatten_list_with_nan(potential_modified_reading_passages_vandalism[key])]

    c122, c133 = [], []

    for item in c12:
      if '\r' in item:
        c122.append(item.replace('\r', '.'))
      else:
        c122.append(item)

    for item in c13:
      if '\r' in item:
        c133.append(item.replace('\r', '.'))
      else:
        c133.append(item)

    for index in range(len(c1)):
      # data['original_full'].append(c1[index])
      # data['modified_full'].append(c2[index])
      # data['changes'].append(c3[index])
      data['time'].append(c4[index])
      data['user'].append(c5[index])
      data['size'].append(c6[index])
      data['comment'].append(c7[index])
      data['tags'].append(c8[index])
      # data['original_pure'].append(c9[index])
      # data['modified_pure'].append(c10[index])
      # data['changes_pure'].append(c11[index])
      data['original passages'].append(c122[index])
      data['modified_passages'].append(c133[index])
    data['title'].extend([''] * (len(c1) - 1))

  df = pd.DataFrame(data)
  print(len(df))
  print("Above for verification!")
  df.to_csv(csv_filename, index=False)

def extract_original_and_modified_reading_passages_vandalism(item):

  processed_page_titles, average_percentage_add, average_percentage_delete, average_percentage_operating = [], [], [], []

  for page_title in item[1]:

    print(page_title)

    mcp, ocp, mc, oc, t, u, s, c, time, inf, inf_p, potential_original_reading_passages, potential_modified_reading_passages_vandalism = [{} for _ in range(13)]

    modified_contents_pure, original_contents_pure, modified_contents, original_contents, timestamp, user, size, comment, tags, info, info_pure, percentage_add, percentage_delete, percentage_operating, original_reading_passages, modified_reading_passages = [[] for _ in range(16)]

    all_versions_data = get_all_versions_data(page_title)

    if all_versions_data != "Error: 'revisions'. 'revisions' key not found in the response data.":

      print("extracted successfully!")
      print()
      processed_page_titles.append(page_title)

      all_versions_data_pure_context = [wikitextparser.parse(avd['content']).plain_text() for avd in all_versions_data]

      print(len(all_versions_data_pure_context))
      print(len(all_versions_data))
      print()

      for version_id in range(len(all_versions_data)-1):

        current_version_pure_text = all_versions_data_pure_context[version_id]
        preceding_version_pure_text = all_versions_data_pure_context[version_id+1]

        if current_version_pure_text.split('\n') != preceding_version_pure_text.split('\n'):

          result = vandalism_pattern_analysis(preceding_version_pure_text.split('\n'), current_version_pure_text.split('\n'))

          original_content_list = result[1]
          modified_content_list = result[2]

          add_pattern_index = [index for index in range(len(original_content_list)) if len(original_content_list[index])==0 and len(modified_content_list[index])>0]
          delete_pattern_index = [index for index in range(len(original_content_list)) if len(original_content_list[index])>0 and len(modified_content_list[index])==0]
          operating_pattern_index = [index for index in range(len(original_content_list)) if len(original_content_list[index])>0 and len(modified_content_list[index])>0]

          if len(original_content_list) != 0:
            percentage_add.append(len(add_pattern_index)/len(original_content_list))
            percentage_delete.append(len(delete_pattern_index)/len(original_content_list))
            percentage_operating.append(len(operating_pattern_index)/len(original_content_list))
          else:
            percentage_add.append(int(0))
            percentage_delete.append(int(0))
            percentage_operating.append(int(0))

          original_operating = [original_content_list[id] for id in operating_pattern_index]
          modified_operating = [modified_content_list[id] for id in operating_pattern_index]

          if len(original_operating)==len(modified_operating) and len(original_operating)!=0 and len(modified_operating)!=0:
            kept_indices_one = [index for index in range(len(original_operating)) if isinstance(original_operating[index], str) and isinstance(modified_operating[index], str)]
            if len(kept_indices_one)>0:
              original_operating_one = [original_operating[id] for id in kept_indices_one]
              modified_operating_one = [modified_operating[id] for id in kept_indices_one]
              kept_indices_two = [index for index in range(len(original_operating_one)) if len(original_operating_one[index])>500 and len(modified_operating_one[index])>500]
              if len(kept_indices_two)>0:

                original_reading_passages.append([original_operating_one[index] for index in kept_indices_two])
                modified_reading_passages.append([modified_operating_one[index] for index in kept_indices_two])

                modified_contents_pure.append(current_version_pure_text)
                original_contents_pure.append(preceding_version_pure_text)

                modified_contents.append(all_versions_data[version_id]['content'])
                original_contents.append(all_versions_data[version_id+1]['content'])

                timestamp.append(all_versions_data[version_id]['timestamp']+"££££££"+all_versions_data[version_id+1]['timestamp'])
                user.append(all_versions_data[version_id]['user'])
                size.append(all_versions_data[version_id]['size']-all_versions_data[version_id+1]['size'])
                comment.append(all_versions_data[version_id]['comment'])
                tags.append(all_versions_data[version_id]['tags'])

                info.append('\n\n'.join(vandalism_pattern_analysis(all_versions_data[version_id+1]['content'].split('\n'), all_versions_data[version_id]['content'].split('\n'))[0]))

                info_pure.append('\n\n'.join(result[0]))

      average_percentage_add.append(round(np.mean(percentage_add)*100, 2))
      average_percentage_delete.append(round(np.mean(percentage_delete)*100, 2))
      average_percentage_operating.append(round(np.mean(percentage_operating)*100, 2))

      print(processed_page_titles)
      print(average_percentage_add)
      print(average_percentage_delete)
      print(average_percentage_operating)

      mcp[f'{page_title}'] = modified_contents_pure
      ocp[f'{page_title}'] = original_contents_pure
      mc[f'{page_title}'] = modified_contents
      oc[f'{page_title}'] = original_contents
      time[f'{page_title}'] = timestamp
      u[f'{page_title}'] = user
      s[f'{page_title}'] = size
      c[f'{page_title}'] = comment
      t[f'{page_title}'] = tags
      inf[f'{page_title}'] = info
      inf_p[f'{page_title}'] = info_pure

      potential_original_reading_passages[f'{page_title}'] = original_reading_passages
      potential_modified_reading_passages_vandalism[f'{page_title}'] = modified_reading_passages

      if len(all_versions_data)-1 != 0:
        print((len(modified_contents_pure)/(len(all_versions_data)-1))*100)
      else:
        print("!")

      if len(original_reading_passages) != 0:
        save_to_csv(mcp, ocp, mc, oc, t, u, s, c, time, inf, inf_p, potential_original_reading_passages, potential_modified_reading_passages_vandalism, f'./{item[0]}/{page_title}.csv')

  return [processed_page_titles, average_percentage_add, average_percentage_delete, average_percentage_operating]

# extract unique Wikipedia article titles from the development set of each MRC dataset (the following is an example on BoolQ)

import json
import random

def read_jsonl_file(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

filepath = './dev.jsonl' # path of the development set of BoolQ
json_data = read_jsonl_file(filepath)

train_page_titles_new_new = []
for id in range(len(json_data)):
  train_page_titles_new_new.append(json_data[id]['title'])
train_page_titles_new = list(set(train_page_titles_new_new))

import os

folder_path = './boolq' # this is the defined folder to save the extracted csv files (each stores the Wikipedia revision histories for one article)

page_titles_dict = {'boolq': train_page_titles_new}

for item in page_titles_dict.items():
  print(f'{item[0]}')
  r = extract_original_and_modified_reading_passages_vandalism(item)
  print()
  print()

  processed_page_titles = r[0]
  average_percentage_add = r[1]
  average_percentage_delete = r[2]
  average_percentage_operating = r[3]

  print(processed_page_titles)
  print(average_percentage_add)
  print(average_percentage_delete)
  print(average_percentage_operating)
  print()

  temporary = [average_percentage_add, average_percentage_delete, average_percentage_operating]

  for index in range(len(temporary)):
    focus = generate_sublists(temporary[index])
    statistics = [round((len(f)/len(temporary[index]))*100, 2) for f in focus]
    print(statistics)
    print()
    print()

# combine the extracted csv files to a single csv file
import pandas as pd
import os

def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

csv_files = get_file_paths('./boolq')

dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

result = pd.concat(dfs, ignore_index=True)

result.to_csv('./boolq.csv', index=False)