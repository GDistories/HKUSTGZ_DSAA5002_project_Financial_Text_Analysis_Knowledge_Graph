#!/usr/bin/env python
# coding: utf-8

# # Task 2: Application of Knowledge Graph

# ## Question 3: Constructing a Knowledge Graph
# 
# 在这一部分，我们将使用Neo4j图数据库来根据[KnowledgeGraph文件夹](../KnowledgeGraph)下的数据构建一个知识图谱。
# 我们可以看到，这个知识图谱包含了公司节点和公司之间的关系，其中公司节点的属性包括公司名称、公司代码和公司标签，公司之间的关系包括竞争、合作、纠纷、投资、同行业和供应关系。
# 
# 以下是我们将要使用的数据文件：
# - [hidy.nodes.company.csv](../KnowledgeGraph/hidy.nodes.company.csv): 公司节点
# - [hidy.relationships.compete.csv](../KnowledgeGraph/hidy.relationships.compete.csv): 竞争关系
# - [hidy.relationships.cooperate.csv](../KnowledgeGraph/hidy.relationships.cooperate.csv): 合作关系
# - [hidy.relationships.dispute.csv](../KnowledgeGraph/hidy.relationships.dispute.csv): 纠纷关系
# - [hidy.relationships.invest.csv](../KnowledgeGraph/hidy.relationships.invest.csv): 投资关系
# - [hidy.relationships.same_industry.csv](../KnowledgeGraph/hidy.relationships.same_industry.csv): 同行业关系
# - [hidy.relationships.supply.csv](../KnowledgeGraph/hidy.relationships.supply.csv): 供应关系
# 
# In this part, we will use Neo4j graph database to construct a knowledge graph based on the data under the [KnowledgeGraph folder](../KnowledgeGraph).
# We can see that this knowledge graph contains company nodes and relationships between companies, where the attributes of company nodes include company name, company code and company label, and the relationships between companies include competition, cooperation, dispute, investment, same industry and supply relationships.
# 
# The following are the data files we will use:
# - [hidy.nodes.company.csv](../KnowledgeGraph/hidy.nodes.company.csv): Company nodes
# - [hidy.relationships.compete.csv](../KnowledgeGraph/hidy.relationships.compete.csv): Compete relationships
# - [hidy.relationships.cooperate.csv](../KnowledgeGraph/hidy.relationships.cooperate.csv): Cooperate relationships
# - [hidy.relationships.dispute.csv](../KnowledgeGraph/hidy.relationships.dispute.csv): Dispute relationships
# - [hidy.relationships.invest.csv](../KnowledgeGraph/hidy.relationships.invest.csv): Invest relationships
# - [hidy.relationships.same_industry.csv](../KnowledgeGraph/hidy.relationships.same_industry.csv): Same industry relationships
# - [hidy.relationships.supply.csv](../KnowledgeGraph/hidy.relationships.supply.csv): Supply relationships

# 首先，我们连接到Neo4j数据库，然后读取数据文件。

# In[8]:


import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

# 这一部分，请根据自己的neo4j数据库的用户名和密码来修改
# Please modify the username and password of your neo4j database
uri = "neo4j://localhost:7687"  
username = "neo4j"
password = "912699176"

driver = GraphDatabase.driver(uri, auth=(username, password))

companies_path = '../KnowledgeGraph/hidy.nodes.company.csv'
relationship_compete_path = '../KnowledgeGraph/hidy.relationships.compete.csv'
relationship_cooperate_path = '../KnowledgeGraph/hidy.relationships.cooperate.csv'
relationship_dispute_path = '../KnowledgeGraph/hidy.relationships.dispute.csv'
relationship_invest_path = '../KnowledgeGraph/hidy.relationships.invest.csv'
relationship_same_industry_path = '../KnowledgeGraph/hidy.relationships.same_industry.csv'
relationship_supply_path = '../KnowledgeGraph/hidy.relationships.supply.csv'

companies_df = pd.read_csv(companies_path)
relationship_compete_df = pd.read_csv(relationship_compete_path)
relationship_cooperate_df = pd.read_csv(relationship_cooperate_path)
relationship_dispute_df = pd.read_csv(relationship_dispute_path)
relationship_invest_df = pd.read_csv(relationship_invest_path)
relationship_same_industry_df = pd.read_csv(relationship_same_industry_path)
relationship_supply_df = pd.read_csv(relationship_supply_path)


# 然后，我们可以查看一下数据的基本信息。
# 
# Then, we can take a look at the basic information of the data.

# In[9]:


columns_info = {
    "Companies Columns": companies_df.columns.tolist(),
    "Compete Relationship Columns": relationship_compete_df.columns.tolist(),
    "Cooperate Relationship Columns": relationship_cooperate_df.columns.tolist(),
    "Dispute Relationship Columns": relationship_dispute_df.columns.tolist(),
    "Invest Relationship Columns": relationship_invest_df.columns.tolist(),
    "Same Industry Relationship Columns": relationship_same_industry_df.columns.tolist(),
    "Supply Relationship Columns": relationship_supply_df.columns.tolist()
}

columns_info


# 

# 我们可以看到，公司节点包含了公司名称、公司代码和公司标签，而公司之间的关系可以分为有时间和无时间两种，其中有时间的关系包括竞争、合作、纠纷和同行业关系，而无时间的关系包括投资和供应关系
# 
# 接下来，我们将使用Neo4j图数据库来构建知识图谱
# 我们先创建公司节点
# 
# We can see that the company node contains the company name, company code and company label, and the relationships between companies can be divided into two types: with time and without time. The relationships with time include competition, cooperation, dispute and same industry relationships, while the relationships without time include investment and supply relationships.
# 
# Next, we will use Neo4j graph database to construct a knowledge graph
# We first create company nodes

# In[3]:


def create_company_node(tx, id, name, code, label):
    query = (
        "MERGE (:Company {id: $id, name: $name, code: $code, label: $label})"
    )
    tx.run(query, id=id, name=name, code=code, label=label)

# create company node
with driver.session() as session:
    for index, row in tqdm(companies_df.iterrows(), total=companies_df.shape[0], desc="Company Nodes"):
        session.execute_write(create_company_node, row[':ID'], row['company_name'], row['code'], row[':LABEL'])


# 然后，我们创建公司之间的关系
# 
# Then, we create the relationships between companies

# In[5]:


def create_dynamic_relation(tx, start_id, end_id, relation_type, time):
    query = (
        f"MATCH (a:Company {{id: $start_id}}), (b:Company {{id: $end_id}}) "
        f"MERGE (a)-[:{relation_type} {{time: $time}}]->(b)"
    )
    tx.run(query, start_id=start_id, end_id=end_id, time=time)

def create_dynamic_relation_without_time(tx, start_id, end_id, relation_type):
    query = (
        f"MATCH (a:Company {{id: $start_id}}), (b:Company {{id: $end_id}}) "
        f"MERGE (a)-[:{relation_type}]->(b)"
    )
    tx.run(query, start_id=start_id, end_id=end_id)


# 对于竞争、合作、纠纷和同行业关系，它们都有时间列，所以我们可以使用create_dynamic_relation函数来创建关系。
# 
# 但是对于投资和供应关系，它们没有时间列，所以我们需要使用create_dynamic_relation_without_time函数来创建关系。
# 
# For compete, cooperate, dispute and same_industry relationship, they have the time column, so we can use the create_dynamic_relation function to create the relationship. 
# 
# But for the invest and supply relationship, they don't have the time column, so we need to use the create_dynamic_relation_without_time function to create the relationship.

# In[6]:


with driver.session() as session:
    # Create compete relationship
    for index, row in tqdm(relationship_compete_df.iterrows(), total=relationship_compete_df.shape[0], desc="Compete Relationships"):
        session.execute_write(create_dynamic_relation, row[':START_ID'], row[':END_ID'], row[':TYPE'], row['time'])
    
    # Create cooperate relationship
    for index, row in tqdm(relationship_cooperate_df.iterrows(), total=relationship_cooperate_df.shape[0], desc="Cooperate Relationships"):
        session.execute_write(create_dynamic_relation, row[':START_ID'], row[':END_ID'], row[':TYPE'], row['time'])
    
    # Create dispute relationship
    for index, row in tqdm(relationship_dispute_df.iterrows(), total=relationship_dispute_df.shape[0], desc="Dispute Relationships"):
        session.execute_write(create_dynamic_relation, row[':START_ID'], row[':END_ID'], row[':TYPE'], row['time'])
        
    # Create same_industry relationship
    for index, row in tqdm(relationship_same_industry_df.iterrows(), total=relationship_same_industry_df.shape[0], desc="Same Industry Relationships"):
        session.execute_write(create_dynamic_relation, row[':START_ID'], row[':END_ID'], row[':TYPE'], row['time'])


# In[7]:


with driver.session() as session:
    # Create invest relationship
    for index, row in tqdm(relationship_invest_df.iterrows(), total=relationship_invest_df.shape[0], desc="Invest Relationships"):
        session.execute_write(create_dynamic_relation_without_time, row[':START_ID'], row[':END_ID'], row[':TYPE'])
        
    # Create supply relationship
    for index, row in tqdm(relationship_supply_df.iterrows(), total=relationship_supply_df.shape[0], desc="Supply Relationships"):
        session.execute_write(create_dynamic_relation_without_time, row[':START_ID'], row[':END_ID'], row[':TYPE'])


# 我们已经完成了知识图谱的构建，我们可以在Browser中查看一下知识图谱的效果
# 这张图展示了全部的知识图谱
# <img src="../Image/Knowledge Graph_Show All.jpg" width="1600" height="1000" align="bottom"/>
# 
# 这张图展示了300条关系的知识图谱
# <img src="../Image/Knowledge Graph_LIMIT 300.jpg" width="1600" height="1000" align="bottom"/>
# 
# 
# We have completed the construction of the knowledge graph, we can take a look at the effect of the knowledge graph in the Browser
# This picture shows all the knowledge graphs
# <img src="../Image/Knowledge Graph_Show All.jpg" width="1600" height="1000" align="bottom"/>
# 
# This picture shows the knowledge graph of 300 relationships
# <img src="../Image/Knowledge Graph_LIMIT 300.jpg" width="1600" height="1000" align="bottom"/>

# 至此，Question 3完成
# 
# So far, Question 3 is completed

# ## Question 4: Knowledge-Driven Financial Analysis

# 在这一部分，我们使用了之前[Task 1](./Task1.ipynb)中的数据，我们将识别出每个Explicit_Company对应的所有Implicit_Company，将它们分为Implicit_Positive_Company和Implicit_Negative_Company。
# 
# 以下是我们将要使用或生成的数据文件：
# - [Task1.xlsx](../Submission_Excel/Task1.xlsx): Task 1的输出文件
# - [Task2.xlsx](../Submission_Excel/Task2.xlsx): Task 2的输出文件
# 
# 
# In this part, we used the data in [Task 1](./Task1.ipynb) before, we will identify ALL implicit companies corresponding to each company
# of Explicit_Company in your own Task1.xlsx file. Categorize them into Implicit Positive Companies and Implicit Negative
# Companies.
# 
# The following are the data files we will use or generate:
# - [Task1.xlsx](../Submission_Excel/Task1.xlsx): Output file of Task 1
# - [Task2.xlsx](../Submission_Excel/Task2.xlsx): Output file of Task 2

# In[2]:


import pandas as pd
df = pd.read_excel("../Submission_Excel/Task1.xlsx")


# In[3]:


negative_relations = ['compete', 'dispute']
positive_relations = ['cooperate', 'invest', 'same_industry', 'supply']  # 根据实际情况调整

df['Implicit_Positive_Company'] = None
df['Implicit_Negative_Company'] = None

def fetch_relationships(tx, company_name):
    query = (
        "MATCH (a:Company {name: $company_name})-[r]->(b) "
        "RETURN type(r) as relation_type, b.name as company_name"
    )
    result = tx.run(query, company_name=company_name)
    return [(record["relation_type"], record["company_name"]) for record in result]

with driver.session() as session:
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): 
        companies = row['Explicit_Company'].split(' ')
        implicit_negative = []
        implicit_positive = []

        for company in companies:
            relationships = session.execute_read(fetch_relationships, company)

            for relation_type, related_company in relationships:
                if row['label'] == 1:
                    if relation_type in negative_relations:
                        implicit_negative.append(related_company)
                    elif relation_type in positive_relations:
                        implicit_positive.append(related_company)
                else:
                    if relation_type in positive_relations:
                        implicit_negative.append(related_company)
                    elif relation_type in negative_relations:
                        implicit_positive.append(related_company)
                        
        df.at[index, 'Implicit_Positive_Company'] = ' '.join(implicit_positive) if implicit_positive else 'None'
        df.at[index, 'Implicit_Negative_Company'] = ' '.join(implicit_negative) if implicit_negative else 'None'
        


# In[4]:


df.to_excel('../Submission_Excel/Task2.xlsx', index=False)


# 我们成功地将每个Explicit_Company对应的所有Implicit_Company识别出并保存。我们可以用下面的代码来查看一下数据的基本信息
# 
# We successfully identified all Implicit_Company corresponding to each Explicit_Company and saved them. We can use the following code to view the basic information of the data

# In[5]:


df


# In[6]:


df.describe(include='all')


# 至此，Question 4完成
# 
# So far, Question 4 is completed

# In[ ]:




