# -*- coding: utf-8 -*-
"""calculate_item_profile.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nxNU6hwyr8zU_U_FJUHzqS3F595rHIUy
"""

import json

file_path = 'data.json'

# with open(file_path,'w', encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False)

#load data

with open(file_path, "r", encoding="utf-8") as json_file:
    data_dict = json.load(json_file)
    print(data_dict)

#item profile calculation

for index,data in enumerate(data_dict):

    profile = [] #item profile
    
    data_fields = data['fields'] #data fields

    ##### genre
    #### 0번 index는 genre의 0번째 원소값을 사용

    #print(data_fields)

    genres = data_fields['genre_ids']

    if len(genres) == 0:
        
        profile.append(20)
    
    else:
        
        profile.append(genres[0])

    ##### popularity
    #### 너무 큰 값이 계산에 큰 영향을 주지 않도록 범주형으로 변환

    popularity = data_fields['popularity']

    if popularity <= 250.0:
        
        profile.append(0)
    
    elif popularity <= 600.0:
        
        profile.append(1)
    
    elif popularity <= 1000.0:
        
        profile.append(2)

    else:
        
        profile.append(3)
    
    #### vote_average
    #### 실수값 그대로 사용

    vote_average = data_fields['actual_vote_average']

    profile.append(vote_average)

    #### running_time
    ### 적절한 구간마다 변수변환

    running_time = data_fields['running_time']

    if running_time <= 100:
        
        profile.append(0)
    
    elif running_time <= 140:
        
        profile.append(1)
    
    else:
        
        profile.append(2)
    

    ### 국가

    nation = data_fields['nation']

    if nation == '한국':
        
        profile.append(0)
    
    elif nation == '미국':
        
        profile.append(1)
    
    elif nation == '일본':
        
        profile.append(2)
    
    else:
        
        profile.append(3)
    
    ##등급

    profile.append(data_fields['grade'])
    
    data_dict[index]['fields']['profile'] = profile

print(data_dict)

file_path = 'vectordata.json'

with open(file_path,'w', encoding="utf-8") as f:
    json.dump(data_dict, f, ensure_ascii=False)

print(data_dict)

