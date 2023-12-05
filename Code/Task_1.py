#!/usr/bin/env python
# coding: utf-8

# # Task 1: Data Preprocessing and Analysis

# ## Question 1: Data Preprocessing - Noise Removal
# Overall:
# <img src="../Image/Data Filtering Flowchart.png" width="1000" height="2500" align="bottom"/>

# ### Part 1: Data Preprocessing
# 在这一部分，我们将对新闻数据进行清洗。
# 以下是在这一部分使用或生成的文件：
# - [News.xlsx](../Raw_data/News.xlsx): 原始新闻数据
# - [A_share_list.json](../Raw_data/A_share_list.json): 股票公司信息
# - [News_Cleaned.xlsx](../Backup/News_Cleaned.xlsx): 清洗后的新闻数据
# 
# In this part, we will clean the news data.
# The following are the files used or generated in this part:
# - [News.xlsx](../Raw_data/News.xlsx): Original news data
# - [A_share_list.json](../Raw_data/A_share_list.json): Stock company information
# - [News_Cleaned.xlsx](../Backup/News_Cleaned.xlsx): Cleaned news data

# 首先，我们读取所有新闻
# First, we read all the news

# In[1]:


import pandas as pd

# 读取原始Excel文件，并将NewsID列作为索引
# Read the original Excel file and set the NewsID column as the index
df = pd.read_excel('News.xlsx', index_col='NewsID')


# 我们使用describe()函数来查看数据集的一些统计信息
# 
# We use the describe() function to view some statistical information about the dataset

# In[2]:


df.describe()


# 新闻来源列是不必要的，因为我们只关心新闻的内容。因此，我们将其删除
# 
# The NewsSource column is unnecessary because we only care about the content of the news. So we drop it.

# In[3]:


# Drop the NewsSource column
df = df.drop(columns=['NewsSource'])


# 接下来，我们查看一下数据集中是否有空值
# 
# Next, we check if there are any null values in the dataset

# In[4]:


# Check for null values
null_values = df.isnull().sum()
null_values


# 我们可以看到，有一些新闻没有内容。因为我们只关心有标题和内容的新闻，所以我们将删除这些新闻
# 
# We can see that some news do not have content. Because we only care about news with titles and content, we will delete these news

# In[5]:


df_drop_null = df.dropna(subset=['Title', 'NewsContent'])
null_values = df_drop_null.isnull().sum()
null_values


# In[6]:


df_drop_null.describe()


# 可以看到，我们删除了421条有空值的新闻
# 接下来，我们删除新闻为全英文的新闻，因为我们只关心中文新闻
# 
# We can see that we deleted 421 news with null values
# Next, we delete the news that are all English, because we only care about Chinese news

# In[7]:


# Drop news that are all English
df_drop_english = df_drop_null[df_drop_null['NewsContent'].str.contains('[\u4e00-\u9fa5]')]
df_drop_english = df_drop_english[df_drop_english['Title'].str.contains('[\u4e00-\u9fa5]')]
df_drop_english.describe()


# 可以看到，我们删除了1036614-1027482=9132条全英文的新闻
# 接下来，我们对新闻的内容进行一些清洗, 删除特殊字符和数字
# 
# We can see that we deleted 1036614-1027482=9132 news that are all English
# Next, we clean the content of the news, delete special characters and numbers

# In[8]:


import re
def preprocess_text(text):
    # 清洗文本，移除特殊字符和数字
    # Clean the text, remove special characters and numbers
    text = re.sub(r'\W+|\d+', ' ', text)
    return text

df_drop_english['ProcessedNewsTitle'] = df_drop_english['Title'].apply(preprocess_text)
df_drop_english['ProcessedNewsContent'] = df_drop_english['NewsContent'].apply(preprocess_text)
df_processed_text = df_drop_english
df_processed_text.describe()


# 在进行新闻文本的预处理后，我们对预处理后的结果的内容进行去重
# 
# After preprocessing the news text, we remove the duplicates in the processed text

# In[9]:


df_drop_duplicate = df_processed_text.drop_duplicates(subset='ProcessedNewsContent')
df_drop_duplicate.describe()


# 可以看到，去重后，我们删除了1027482-993503=33979条重复的新闻
# 接下来，我们将新闻标题和新闻内容合并到一列中，得到MergedContent列
# 
# We can see that after removing the duplicates, we deleted 1027482-993503=33979 duplicate news
# Next, we merge the news title and news content into one column to get the MergedContent column

# In[10]:


df_drop_duplicate['MergedContent'] = (
    df_drop_duplicate['ProcessedNewsTitle'].str.replace(r'\s+', ' ', regex=True) + " " +
    df_drop_duplicate['ProcessedNewsContent'].str.replace(r'\s+', ' ', regex=True)
)
df_drop_duplicate.describe()


# 处理完数据后，我们首先将处理后的新闻数据保存到Excel文件中，以便后续使用
# 
# After processing the data, we first save the processed news data to an Excel file for later use

# In[11]:


df_drop_duplicate.to_excel('News_Cleaned.xlsx')


# 至此，我们完成了新闻数据的清洗
# At this point, we have completed the cleaning of the news data

# ### Part 2: News Data Categorization: Full Match
# 
# 在这一部分，我们通过完全匹配的方式，将新闻数据分为完整相关和其他的新闻。
# 以下是在这一部分使用或生成的文件：
# - [A_share_list.json](../Raw_data/A_share_list.json): 股票公司信息
# - [News_Cleaned.xlsx](../Backup/News_Cleaned.xlsx): 清洗后的新闻数据
# - [Full_Relevant_News.xlsx](../Backup/Full_Relevant_News.xlsx): 完全匹配A_share_list.json中的name、fullname、code的新闻
# - [Other_News.xlsx](../Backup/Other_News.xlsx): 其他新闻（不完全匹配A_share_list.json中的name、fullname、code的新闻）
# 
# In this part, we categorize the news data into full relevant and other news by full matching.
# The following are the files used or generated in this part:
# - [A_share_list.json](../Raw_data/A_share_list.json): Stock company information
# - [News_Cleaned.xlsx](../Backup/News_Cleaned.xlsx): Cleaned news data
# - [Full_Relevant_News.xlsx](../Backup/Full_Relevant_News.xlsx): News that fully match the name, fullname, or code in A_share_list.json
# - [Other_News.xlsx](../Backup/Other_News.xlsx): Other news (news that do not fully match the name, fullname, or code in A_share_list.json)
# 

# 首先，我们读取A_share_list.json和News_Cleaned.xlsx中的数据
# 
# First, we read the data in A_share_list.json and News_Cleaned.xlsx

# In[1]:


import json
import pandas as pd
import re
from tqdm import tqdm

# 从JSON文件中读取股票信息
with open('A_share_list.json', 'r', encoding='utf-8') as file:
    share_list = json.load(file)

# 创建字典以存储不同类型的关键字
keywords = {'name': {}, 'fullname': set(), 'code': set()}

# 定义正则表达式，删除股票名称中的特定字符
patterns_to_remove = [r'\*?ST', r'^PT', r'^S', r'B股$', r'B$', r'A$']

for item in share_list:
    cleaned_name = item['name']
    for pattern in patterns_to_remove:
        cleaned_name = re.sub(pattern, '', cleaned_name)

    # 存储清理后的名称与原始名称的映射
    keywords['name'][cleaned_name] = item['name']
    keywords['fullname'].add(item['fullname'])
    keywords['code'].add(item['code'])

# 读取新闻数据
news_df = pd.read_excel('News_Cleaned.xlsx', index_col='NewsID')


# 我们首先使用完全匹配的方式，将新闻数据根据是否完全匹配json文件中的name、fullname、code，分为完整相关和其他的新闻
# 
# First, we use full matching to categorize the news data into full relevant and other news according to whether they fully match the name, fullname, or code in the json file

# In[2]:


# 定义函数来检查新闻标题或内容中是否包含特定类型的关键字
def contains_keyword(row, keyword_type):
    matches = []
    if keyword_type == 'name':
        for cleaned_name, original_name in keywords[keyword_type].items():
            if cleaned_name in row['Title'] or cleaned_name in row['NewsContent']:
                matches.append(original_name)
    else:
        for keyword in keywords[keyword_type]:
            if keyword in row['Title'] or keyword in row['NewsContent']:
                matches.append(keyword)
    return matches if matches else ''

# 应用进度条
tqdm.pandas(desc="Processing news data")

# 对每种类型进行检查
for key_type in ['name', 'fullname', 'code']:
    news_df[key_type] = news_df.progress_apply(contains_keyword, axis=1, args=(key_type,))


# 接下来，我们确定哪些新闻是相关的（即至少匹配一个关键字类型），并将其保存到Excel文件中
# 
# Next, we determine which news are relevant (i.e. match at least one keyword type) and save them to an Excel file

# In[3]:


# 确定哪些新闻是相关的（即至少匹配一个关键字类型）
# Determine which news are relevant (i.e. match at least one keyword type)
news_df['relevant'] = news_df[['name', 'fullname', 'code']].any(axis=1)
complete_matching_news = news_df[news_df['relevant']]
incomplete_matching_news = news_df[~news_df['relevant']]


# In[4]:


# 保存完全匹配的新闻到Excel文件中
# Save the fully matching news to an Excel file
complete_matching_news.to_excel('Full_Relevant_News.xlsx')
complete_matching_news.describe()


# In[5]:


# 保存其他新闻到Excel文件中
# Save the other news to an Excel file
incomplete_matching_news.to_excel('Other_News.xlsx')
incomplete_matching_news.describe()


# 至此，我们通过完全匹配的方式，将新闻数据分为完整相关和其他的新闻。
# At this point, we divide the news data into fully relevant and other news through a perfect match.

# ### Part 3: News Data Categorization: Similarity Matching

# 在这一部分，由于新闻中的股票名称可能会出现不同的写法，例如“中国平安”和“平安”，因此我们需要对这一部分的新闻进行进一步地处理，提取其中的公司名称
# 
# 经过抽样100条运行并人工复核，使用Bert以及计算余弦相似度得到的提取结果比较差。
# 如果您对该部分的代码感兴趣，可以在这里找到：
# [Deprecated Bert Code](../Deprecated/bert.ipynb) 
# 
# 经过探索，我发现了百度公司的LAC工具，可以对中文文本进行词性标注，我们可以通过词性标注来提取新闻中的组织名。
# 因此，我们决定使用LAC工具来提取关键字，并使用thefuzz来计算相似度。LAC工具可以对中文文本进行词性标注，我们可以通过词性标注来提取新闻中的组织名。
# 
# 以下是在这一部分使用或生成的文件：
# - [A_share_list.json](../Raw_data/A_share_list.json): 股票公司信息
# - [Other_News.xlsx](../Backup/Other_News.xlsx): 其他新闻（不完全匹配A_share_list.json中的name、fullname、code的新闻）
# - [Other_news_with_ORGs.xlsx](../Backup/Other_news_with_ORGs.xlsx): 使用LAC工具提取了组织名的其他新闻
# - [Cleaned_Other_news_with_ORGs.xlsx](../Backup/Cleaned_Other_news_with_ORGs.xlsx): 清洗后的使用LAC工具提取了组织名的其他新闻
# - [Cleaned_Other_news_with_ORGs_and_Similarities.xlsx](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx): 使用thefuzz计算相似度后，分数大于80的其他新闻
# 
# In this part, since the stock names in the news may appear in different ways, such as "中国平安" and "平安", we need to further process this part of the news and extract the company names in them.
# 
# After sampling 100 news and manually reviewing them, the results obtained using Bert and calculating the cosine similarity are not very good. 
# If you are interested in the code of this part, you can find it here:
# [Deprecated Bert Code](../Deprecated/bert.ipynb)
# 
# After exploration, I found Baidu's LAC tool, which can perform part-of-speech tagging on Chinese text, and we can use part-of-speech tagging to extract organization names in the news.
# Therefore, we decided to use the LAC tool to extract keywords and use thefuzz to calculate the similarity. The LAC tool can perform part-of-speech tagging on Chinese text, and we can use part-of-speech tagging to extract organization names in the news.
# 
# The following are the files used or generated in this part:
# - [A_share_list.json](../Raw_data/A_share_list.json): Stock company information
# - [Other_News.xlsx](../Backup/Other_News.xlsx): Other news (news that do not fully match the name, fullname, or code in A_share_list.json)
# - [Other_news_with_ORGs.xlsx](../Backup/Other_news_with_ORGs.xlsx): Other news with extracted organization names using the LAC tool
# - [Cleaned_Other_news_with_ORGs.xlsx](../Backup/Cleaned_Other_news_with_ORGs.xlsx): Cleaned other news with extracted organization names using the LAC tool
# - [Cleaned_Other_news_with_ORGs_and_Similarities.xlsx](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx): Other news with a score greater than 80 after calculating the similarity using thefuzz

# 
# 我们首先使用百度公司的LAC对新闻内容进行词性标注，然后提取词性为ORG的词，作为关键字
# LAC工具的代码库可以在这里找到：
# [Baidu/lac](https://github.com/baidu/lac)
# 
# LAC工具的论文可以在这里找到：
# Jiao, Z., Sun, S., & Sun, K. (2018). Chinese Lexical Analysis with Deep Bi-GRU-CRF Network. *arXiv preprint arXiv:1807.01882*. Retrieved from [https://arxiv.org/abs/1807.01882](https://arxiv.org/abs/1807.01882)
# 
# We first use Baidu's LAC to perform part-of-speech tagging on the news content, and then extract the words with the part of speech ORG as keywords.
# The code library of the LAC tool can be found here:
# [Baidu/lac](https://github.com/baidu/lac)
# 
# The paper of the LAC tool can be found here:
# Jiao, Z., Sun, S., & Sun, K. (2018). Chinese Lexical Analysis with Deep Bi-GRU-CRF Network. *arXiv preprint arXiv:1807.01882*. Retrieved from [https://arxiv.org/abs/1807.01882](https://arxiv.org/abs/1807.01882)

# In[6]:


# 读取其他新闻
# Read the other news
other_news = pd.read_excel('Other_News.xlsx', index_col='NewsID')


# In[7]:


# 使用LAC进行处理
# Use LAC for processing
from LAC import LAC
from tqdm import tqdm
import re

lac = LAC(mode='lac')
# 函数：仅输出词性标签为'ORG'的词，并清理非中英文字符
# Function: only output words with part of speech tag 'ORG', and clean non-Chinese and non-English characters
def extract_orgs_and_clean(text):
    lac_result = lac.run([text])
    orgs = set()  # 使用集合来存储组织名，以避免重复 # Use a set to store the organization name to avoid duplication
    for sentence in lac_result:
        words, tags = sentence
        orgs.update([re.sub(r'[^\u4e00-\u9fffA-Za-z]', '', word) for word, tag in zip(words, tags) if tag == 'ORG'])
    return ' '.join(orgs)  # 将结果以空格分隔的字符串形式返回，不会包含重复的组织名 # Return the result as a string separated by spaces, which will not contain duplicate organization names


tqdm.pandas(desc="Processing progress")
other_news['ORGs'] = other_news['MergedContent'].progress_apply(extract_orgs_and_clean)


# In[8]:


# 保存结果到Excel文件中
# Save the result to an Excel file
other_news.to_excel('Other_news_with_ORGs.xlsx')
other_news.describe(include='all')


# In[9]:


# 删除无用列
# Drop the useless columns
cleaned_other_news = other_news.drop(columns=['ProcessedNewsTitle', 'ProcessedNewsContent', 'name', 'fullname', 'code', 'relevant'])

# 删除空行
# Drop the empty rows
cleaned_other_news = cleaned_other_news[cleaned_other_news['ORGs'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

cleaned_other_news.describe(include='all')


# In[10]:


# 保存结果到Excel文件中
# Save the result to an Excel file
cleaned_other_news.to_excel('Cleaned_Other_news_with_ORGs.xlsx')


# 可以看到，我们成功地从其他新闻中提取了组织名，并将其保存到了Excel文件中。
# 接下来，我们使用thefuzz来计算相似度，将相似度大于80的组织名与A_share_list.json中的name进行匹配，得到匹配的组织名。
# 
# We can see that we successfully extracted the organization names from the other news and saved them to an Excel file.
# Next, we use thefuzz to calculate the similarity, match the organization names with a similarity greater than 80 with the name in A_share_list.json, and obtain the matching organization names.

# In[2]:


import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm
import json

cleaned_other_news = pd.read_excel('Cleaned_Other_news_with_ORGs.xlsx', index_col='NewsID')
with open('A_share_list.json', 'r', encoding='utf-8') as file:
    share_list = json.load(file)

name_list = [item['name'] for item in share_list]
fullname_list = [item['fullname'] for item in share_list]

# 计算相似度函数
# Similarity calculation function
def calculate_similarity(orgs, threshold=80):
    scores = []
    names = []
    for name in name_list:
        for org in orgs.split():
            score = fuzz.ratio(org, name)
            if score >= threshold:
                scores.append(score)
                names.append(name)

    # Return empty strings if scores and names are empty
    return scores if scores else '', names if names else ''


# In[3]:


# Apply the modified function with a progress bar
tqdm.pandas(desc="Calculating similarities")
cleaned_other_news['Similarity_Scores'], cleaned_other_news['Matching_Names'] = zip(*cleaned_other_news['ORGs'].progress_apply(calculate_similarity))


# In[5]:


# Remove rows where 'Similarity_Scores' and 'Matching_Names' are empty strings
cleaned_other_news = cleaned_other_news[(cleaned_other_news['Similarity_Scores'] != '') & (cleaned_other_news['Matching_Names'] != '')]


# In[7]:


cleaned_other_news.describe(include='all')


# In[8]:


# 保存结果到Excel文件中
# Save the result to an Excel file
cleaned_other_news.to_excel('Cleaned_Other_news_with_ORGs_and_Similarities.xlsx')


# 至此，我们成功地计算了相似度，并将相似度大于80的组织名与A_share_list.json中的name进行匹配，得到匹配的组织名。
# 
# At this point, we have successfully calculated the similarity, matched the organization names with a similarity greater than 80 with the name in A_share_list.json, and obtained the matching organization names.

# ### Part 4: Data Merging

# 在这一部分，我们将完全匹配的数据[Full_Relevant_News](../Backup/Full_Relevant_News.xlsx)和通过LAC以及thefuzz筛选后的数据[Cleaned_Other_news_with_ORGs_and_Similarities](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx)进行合并，得到一个完整的数据集，来进行情感分析。
# 
# 以下是在这一部分使用或生成的文件：
# - [A_share_list.json](../Raw_data/A_share_list.json): 股票公司信息
# - [Full_Relevant_News.xlsx](../Backup/Full_Relevant_News.xlsx): 完全匹配A_share_list.json中的name、fullname、code的新闻
# - [Cleaned_Other_news_with_ORGs_and_Similarities.xlsx](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx): 使用thefuzz计算相似度后，分数大于80的其他新闻
# - [Merged_News.xlsx](../Backup/Merged_News.xlsx): 两个数据集合并后的完整数据集
# - [Task1.xlsx](../Submission_Excel/Task1.xlsx): Task 1的提交结果
# 
# In this part, we merge the fully matching data [Full_Relevant_News](../Backup/Full_Relevant_News.xlsx) and the data filtered by LAC and thefuzz [Cleaned_Other_news_with_ORGs_and_Similarities](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx) to obtain a complete data set for sentiment analysis.
# 
# The following are the files used or generated in this part:
# - [A_share_list.json](../Raw_data/A_share_list.json): Stock company information
# - [Full_Relevant_News.xlsx](../Backup/Full_Relevant_News.xlsx): News that fully match the name, fullname, or code in A_share_list.json
# - [Cleaned_Other_news_with_ORGs_and_Similarities.xlsx](../Backup/Cleaned_Other_news_with_ORGs_and_Similarities.xlsx): Other news with a score greater than 80 after calculating the similarity using thefuzz
# - [Merged_News.xlsx](../Backup/Merged_News.xlsx): The complete data set after merging the two data sets
# - [Task1.xlsx](../Submission_Excel/Task1.xlsx): The submission result of Task 1

# In[1]:


import pandas as pd
import json
import ast

full_relevant_news = pd.read_excel('Full_Relevant_News.xlsx')
cleaned_other_news = pd.read_excel('Cleaned_Other_news_with_ORGs_and_Similarities.xlsx')

# 从JSON文件中读取股票信息
# Read the stock information from the JSON file
with open('A_share_list.json', 'r', encoding='utf-8') as file:
    share_list = json.load(file)


# In[2]:


full_relevant_news.describe(include='all')


# In[3]:


# 删除无用列
# Drop the useless columns
full_relevant_news = full_relevant_news.drop(['ProcessedNewsTitle', 'ProcessedNewsContent', 'MergedContent'], axis=1)

# 重命名列
# Rename the columns
full_relevant_news = full_relevant_news.rename(columns={'relevant': 'full_relevant'})


# In[4]:


# 创建以 'fullname' 为键，'name' 为值的字典
# Create a dictionary with 'fullname' as the key and 'name' as the value
fullname_map = {item['fullname']: item['name'] for item in share_list if 'fullname' in item}

# 创建以 'code' 为键，'name' 为值的字典
# Create a dictionary with 'code' as the key and 'name' as the value
code_map = {item['code']: item['name'] for item in share_list if 'code' in item}

# 更新 DataFrame
# Update the DataFrame
for index, row in full_relevant_news.iterrows():
    # 初始化或获取当前 name 列的列表
    # Initialize or get the list of the current name column
    current_names = ast.literal_eval(row['name']) if pd.notna(row['name']) else []

    if pd.notna(row['fullname']):
        try:
            fullname_list = ast.literal_eval(row['fullname'])
            if fullname_list and isinstance(fullname_list, list):
                fullname = fullname_list[0]
                if fullname in fullname_map and fullname_map[fullname] not in current_names:
                    current_names.append(fullname_map[fullname])
        except (ValueError, SyntaxError):
            pass

    if pd.notna(row['code']):
        try:
            code_list = ast.literal_eval(row['code'])
            if code_list and isinstance(code_list, list):
                code = code_list[0]
                if code in code_map and code_map[code] not in current_names:
                    current_names.append(code_map[code])
        except (ValueError, SyntaxError):
            pass
    
    full_relevant_news.at[index, 'name'] = str(current_names)


# In[5]:


# 删除无用列
# Drop the useless columns
full_relevant_news = full_relevant_news.drop(['Title', 'fullname', 'code'], axis=1)


# In[6]:


full_relevant_news.describe(include='all')


# 至此，我们完成了完全匹配的新闻数据的处理。下面，我们对清洗后的其他新闻数据进行处理。
# 
# At this point, we have completed the processing of the fully matching news data. Next, we process the cleaned other news data.

# In[7]:


cleaned_other_news.describe(include='all')


# In[8]:


# 删除无用列
# Drop the useless columns
cleaned_other_news = cleaned_other_news.drop(['Title', 'MergedContent', 'ORGs', 'Similarity_Scores'], axis=1)

# 重命名列
# Rename the columns
cleaned_other_news = cleaned_other_news.rename(columns={'Matching_Names': 'name'})

# 添加 'full_relevant' 列，与完全匹配的新闻数据区分
# Add the 'full_relevant' column to distinguish it from the fully matching news data
cleaned_other_news['full_relevant'] = False


# In[9]:


cleaned_other_news.describe(include='all')


# 至此，我们完成了清洗后的其他新闻数据的处理。下面，我们将完全匹配的新闻数据和清洗后的其他新闻数据进行合并，得到一个完整的数据集。
# 
# At this point, we have completed the processing of the cleaned other news data. Next, we merge the fully matching news data and the cleaned other news data to obtain a complete data set.

# In[21]:


merged_news = pd.concat([full_relevant_news, cleaned_other_news], ignore_index=True)


# In[22]:


merged_news.describe(include='all')


# In[23]:


# 保存结果到Excel文件中
# Save the result to an Excel file
merged_news.to_excel('Merged_News.xlsx', index=False)


# 至此，Question 1完成
# 我们删除了原始数据集中的噪音，即任何没有提及任何中国a股上市公司的新闻，得到了583003条新闻。
# 整体过滤率为：583003/1037035=56.2%
# 
# At this point, Question 1 is completed.
# We deleted the noise in the original data set, that is, any news that did not mention any Chinese A-share listed company, and obtained 583003 news.
# The overall filtering rate is: 583003/1037035=56.2%
# 

# ## Question 2: Data Analysis - Text Knowledge Mining

# ### Part 5: Sentiment Analysis
# 
# 由于Colab计算资源的限制，我将文件切为12份，每份48585条新闻（除了最后一个文件），然后使用Colab进行情感分析。
# 
# 其中，情感分析部分使用[IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)进行。
# Colab部分的具体代码可以在这里找到：
# [Part 1](../Sentiment_Analysis/Code/Part_1.ipynb)
# 这里是Part 1的代码。其余代码与Part 1类似，只是读取的文件不同。
# 以下是在这一部分使用或生成的文件：
# - [Merged_News.xlsx](../Backup/Merged_News.xlsx): 两个数据集合并后的完整数据集
# - [Split_data](../Sentiment_Analysis/Split_data): 切分后的数据集(Part 1-12)
# - [Sentiment_Analysis_Results](../Sentiment_Analysis/Result): 情感分析结果(Part 1-12)
# - [Merged_Result.xlsx](../Backup/Merged_Result.xlsx): 合并后的情感分析结果
# 
# Due to the limitation of Colab computing resources, I split the file into 12 parts, each with 48585 news (except for the last file), and then use Colab for sentiment analysis.
# 
# Among them, the sentiment analysis part uses [IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment).
# The specific code of the Colab part can be found here:
# [Part 1](../Sentiment_Analysis/Code/Part_1.ipynb)
# Here is the code for Part 1. The rest of the code is similar to Part 1, except that the files read are different.
# The following are the files used or generated in this part:
# - [Merged_News.xlsx](../Backup/Merged_News.xlsx): The complete data set after merging the two data sets
# - [Split_data](../Sentiment_Analysis/Split_data): The split data set (Part 1-12)
# - [Sentiment_Analysis_Results](../Sentiment_Analysis/Result): Sentiment analysis results (Part 1-12)
# - [Merged_Result.xlsx](../Backup/Merged_Result.xlsx): Merged sentiment analysis results

# In[ ]:


import pandas as pd

# 读取原始Excel文件
df = pd.read_excel('Merged_News.xlsx', index_col='NewsID')


# In[ ]:


# 设定每个分割的大小
split_size = 48585

# 分割数据框
splits = [df[i:i + split_size] for i in range(0, len(df), split_size)]

# 保存每个分割为一个单独的Excel文件
for index, split in enumerate(splits):
    filename = f'Merged_News_Part_{index + 1}.xlsx'
    split.to_excel(filename)


# 我们成功地将数据集切分为12份，每份48585条新闻。（除了最后一个文件）
# 接下来，我们使用Colab的资源进行情感分析。
# 你可以在这里找到在Colab运行的所有代码及代码的运行过程：
# [Sentiment Analysis_Code](../Sentiment_Analysis/Code)
# 
# 在运行完所有代码后，我们得到了12个文件，每个文件包含48585条新闻的情感分析结果。（除了最后一个文件）
# 得到的文件可以在这里找到：
# [Sentiment_Analysis_Results](../Sentiment_Analysis/Result)
# 接下来，我对得到的12个文件进行合并，得到一个完整的数据集。
# 
# We successfully split the dataset into 12 parts, each with 48585 news. (Except for the last file)
# Next, we use Colab's resources for sentiment analysis.
# You can find all the code and the running process of the code running in Colab here:
# [Sentiment Analysis_Code](../Sentiment_Analysis/Code)
# 
# After running all the code, we got 12 files, each containing the sentiment analysis results of 48585 news. (Except for the last file)
# The resulting files can be found here:
# [Sentiment_Analysis_Results](../Sentiment_Analysis/Result)
# Next, I merge the 12 files to get a complete data set.

# In[1]:


import pandas as pd

# 创建一个空的DataFrame来存储合并的数据
# Create an empty DataFrame to store the merged data
merged_df = pd.DataFrame()

# 循环读取每个文件并合并
# Loop through each file and merge
for i in range(1, 13):
    file_name = f"Sentiment_Analysis_Results/Result/Result_{i}.xlsx"  # 构建文件名 # Construct the file name
    print(f"Reading {file_name}...")
    df = pd.read_excel(file_name)   # 读取Excel文件
    merged_df = pd.concat([merged_df, df], ignore_index=True)  # 合并到总的DataFrame # Merge to the total DataFrame


# In[2]:


merged_df.describe(include='all')


# In[3]:


merged_df = merged_df.drop(columns=['Unnamed: 0'])
# 保存结果到Excel文件中
# Save the result to an Excel file
merged_df.to_excel('../Backup/Merged_Result.xlsx', index=False)


# 接下来，我们整理格式，按照提交要求，将结果保存到Excel文件中。
# 
# Next, we organize the format and save the results to an Excel file according to the submission requirements.

# In[4]:


merged_df = merged_df.drop(columns=['full_relevant'])


# In[5]:


merged_df.describe(include='all')


# In[6]:


import ast

merged_df['name'] = merged_df['name'].apply(ast.literal_eval)
merged_df['name'] = merged_df['name'].apply(lambda x: ' '.join(x))


# In[9]:


merged_df = merged_df.rename(columns={'name': 'Explicit_Company'})


# In[10]:


merged_df


# In[11]:


merged_df.describe(include='all')


# In[12]:


# 保存结果到Excel文件中
# Save the result to an Excel file
merged_df.to_excel('Task1.xlsx', index=False)


# 至此，Task 1完成
# 我们成功地从原始数据集中提取了583003条至少提及一个中国a股上市公司的新闻，并对这些新闻进行了情感分析，得到了[Task1.xlsx](../Submission_Excel/Task1.xlsx)文件。
# 
# At this point, Task 1 is completed.
# We successfully extracted 583003 news from the original data set that mentioned at least one Chinese A-share listed company, and performed sentiment analysis on these news, and obtained the [Task1.xlsx](../Submission_Excel/Task1.xlsx) file.
