# Enhancing Text Retrieval with Metadata Filters using MongoDB and LangChain Agent

Retrieving relevant documents based on text similarity can be challenging, especially when users seek information based on specific criteria like dates or categories. Traditional similarity algorithms might not always yield accurate results under these conditions. In this tutorial we will outlines a method to prefilter data using metadata extraction with MongoDB vector search and LangChain Agent, ensuring more precise retrieval of documents.


#### Getting Started
Before diving into the tutorial, ensure you have the following prerequisites:

- A MongoDB Atlas cluster (version 6.0.11, 7.0.2, or later). For setup instructions, visit [Create an Atlas Account - MongoDB Atlas](https://www.mongodb.com/docs/atlas/tutorial/create-atlas-account/) and [Deploy a Free Cluster - MongoDB Atlas](https://www.mongodb.com/docs/atlas/tutorial/deploy-free-tier-cluster/).
    
- An API key from OpenAI or Azure OpenAI. For more information, check out [OpenAI Platform](https://platform.openai.com/docs/quickstart) and [What is Azure OpenAI Service? - Azure AI services | Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview).
    
- Installation of necessary libraries:
    

```
%pip install --upgrade --quiet langchain langchain-mongodb langchain-openai pymongo pypdf
```

### The Dataset 

This tutorial utilizes the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from HuffPost, covering news headlines from 2012 to 2022. Each record includes attributes like category, headline, authors, link, short_description, and date.

### Setting Up

##### 1. Establishing OpenAI Connections
First, create a  OpenAI connection for embedding and completion.

In this article I am going to use Azure OpenAI Models, but OpenAI Models should work also. 

```
from langchain_core.tools import BaseTool, tool
from openai import BaseModel
from pymongo import MongoClient
import os
from typing import Dict, List, Optional, Tuple, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.vectorstores import Neo4jVector, MongoDBAtlasVectorSearch

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="embedding2",
    openai_api_version="2023-05-15",
)
llm = AzureChatOpenAI(
    azure_deployment=<deployment-Name>,
    openai_api_version="2023-05-15",
)
 
client: MongoClient = MongoClient(CONNECTION_STRING)

llm.invoke("hello")
```

#### 2. Index Creation
Next, create an Atlas Vector Search index to  efficient data retrieval based on vector similarity and metadata filters.

The definition will be as the following:

```
{

  "fields": [

    {

      "numDimensions": 1536,

      "path": "embedding",

      "similarity": "cosine",

      "type": "vector"

    },

    {

      "path": "authors",

      "type": "filter"

    },

    {

      "path": "category",

      "type": "filter"

    },

    {

      "path": "date",

      "type": "filter"

    }

  ]

}
```
- A string fields (`category`, `authors`, `date`) for pre-filtering the data.
- The vector embeddings field (`embedding`) for performing vector search against pre-filtered data.

3. Load the data
We can now embed and store into MongoDB by reading the data in JSON format and load it using [DataFrameLoader]([Pandas DataFrame | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/integrations/document_loaders/pandas_dataframe/)) from LangChain, so that we can search over them at runtime.


```
def create_index():  
  
    f=open('dataset.json')  
    df = pd.read_json(f, lines=True)  
    df['page_content']=("link: "+ df["link"] + ",headline " + df["headline"]+ ",authors: " + df["authors"]+ ",category:  "  
                        + df["category"]+ ",short_description:  " + df["short_description"])  
  
 
    docs=DataFrameLoader(df, page_content_column="page_content")  
  
    vectorstore=MongoDBAtlasVectorSearch.from_documents(  
        docs.load(),  
        embeddings,  
        collection=collection,  
        index_name=INDEX_NAME  
    )
```

For more details about creating index, [MongoDB Atlas | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/integrations/vectorstores/mongodb_atlas/)

There is no need in our case to split documents in our case. After that, you can check the collection, and we should see the data in the collection, I am using  
[MongoDB Compass | MongoDB](https://www.mongodb.com/products/tools/compass) for that.



#### 3. Querying the Index

We will start by reading the index that we already created so we can use to query our data.
```
def read_index():  
    DB_NAME= 'prefilter'  
    return  MongoDBAtlasVectorSearch(  
        client[DB_NAME]['prefilter'], embeddings, index_name='vector_index'  
    )
```

 We take a text search query, embed it, and perform some sort of “similarity” search to identify the stored splits with the most similar embeddings to our query embedding. The simplest similarity measure is cosine similarity — we measure the cosine of the angle between each pair of embeddings (which are high dimensional vectors).

``` python
vector_index = read_index()
vector_index.similarity_search_with_score(k= 4,query="give articles talks about Covid")
```


#####  4.Creating  the Data Extraction Tool  

Tools are functions that an agent can invoke. The `Tool` abstraction consists of two components:

1. The input schema for the tool. This tells the LLM what parameters are needed to call the tool. Without this, it will not know what the correct inputs are. These parameters should be sensibly named and described.
2. The function to run. Include the input Schema as prefilter fields before retrieving the data from  the the MongoDB collection.
 First, will start by creating a class for arguments schema  for our extraction tool, and providing  some examples so that the LLM would understand it better, you can observe that we give the LLM information about the format and examples as well as provide an enumeration. 
```
class NewsInput(BaseModel):  
    category: Optional[str] = Field(  
        description="Any particular category that the user wants to finds information for. Here are some examples: "  
        +  """{Input:show me articles about food ? category: food} , {Input: is there any articles tagged U.S. News talking about about Covid ? category: U.S. News"""  
    )  
    authors: Optional[str] = Field(  
        description="the Author  Name that wrote articles the user wants to find articles for "  
        +"""{Input:give article written by Marry lother? Auther: Marry lother}, {input: is Nina Golgowski have any articles? Author:Nina Golgowski """  
    )  
    date: Optional[str] = Field(  
        description="the  date of an article that the use want to use to filter article, rewrite it format yyyy-MM-ddTHH:mm:ss"  
    )  
    determination: Optional[str] = Field(  
        description="the condition for the date that the user want to filter on ", enum=["before", "after","equal"]  
    )  
    desc: Optional[str] = Field(  
        description="the details and description in the article the user is looking  in the article or contained in the article"  
    )
```
 By understanding how the users will use the model, it will help writing a better schema description for the Extraction schema:
 
For example, if the user entered the following prompt:

```json 
{"input": "give me articles written by Elyse Wanshel after 22nd of Sep about Comedy"}
```

The Extraction function will return the argument for the tool as following:
```json
{'authors': 'Elyse Wanshel', 'date': '2022-09-22T00:00:00', 'determination': 'after', 'category': 'Comedy'}
```

Now we can implement the  function to run taking the class we created above as arguments schema

```
@tool(args_schema=NewsInput)  
def get_articles(  
    category: Optional[str] = None,  
    authors: Optional[str] = None,  
    date: Optional[str] = None,  
desc: Optional[str] = None,  
        determination:Optional[str] =None  
) -> str:  
    "useful for when you need to find relevant information in the news"  
    vector_index = read_index()  
  
    condition=''  
    if determination =="before":  
        condition = "$lte"  
    elif determination =="after":  
        condition="$gte"  
    elif determination == "equal":  
        condition="$eq"  
  
    filter ={}  
    if category is not None:  
        filter["category"]= {"$eq": category.upper()}  
    if  authors is not None:  
        filter["authors"] = {"$eq": authors}  
    if date is not None:  
        filter["date"] = {condition:  datetime.fromisoformat(date)}  
  
    return  format_docs(vector_index.similarity_search_with_score(k= 4,query=desc if desc else '', pre_filter = {'$and': [  
        filter ] }))
        
tools = [get_articles]  
      
```

The LangChain will take the arguments for the similarity_search_with_score and create the following query for the MongoDB 

```
{'queryVector': [0.001553418948702656, -0.016994878857730846,....], 'path': 'embedding', 'numCandidates': 40, 'limit': 4, 'index': 'vector_index', 'filter': {' $and': [{'category': {'$eq': 'COMEDY'}, 'authors': {' $eq': 'Elyse Wanshel'}, 'date': {'$gte': datetime.Datetime (2022, 9, 22, 0, 0)}}]}}
```

##### 5.Create Agent 

We need now to create Agent, Agent use OpenAI model to decide if it need to call the tool. They require an `executor`, which is the runtime for the agent. The executor is what actually calls the agent, executes the tools it chooses, passes the action outputs back to the agent, and repeats. The agent is responsible for parsing output from the previous results and choosing the next steps.


we first create the prompt we want to use to guide the agent.

```

  
prompt = ChatPromptTemplate.from_messages(  
    [  
        (  
            "system",  
            "You are a helpful assistant that finds information about articles "  
            "make sure to ask the user for clarification. Make sure to include any "            "available options that need to be clarified in the follow up questions "            "Do only the things the user specifically requested. ",  
        ),  
        MessagesPlaceholder(variable_name="chat_history"),  
        ("user", "{input}"),  
        MessagesPlaceholder(variable_name="agent_scratchpad"),  
    ]  
)  
```


We can initialize the agent with the  OpenAI, the prompt, and the tools. The agent is responsible for taking in input and deciding what actions to take. Crucially, the Agent does not execute those actions that is done by the AgentExecutor.



```python
  
from langchain.agents import AgentExecutor, create_tool_calling_agent  
agent = create_tool_calling_agent(llm, tools, prompt)  

```
Finally, we combine the agent  with the tools inside the AgentExecutor (which will repeatedly call the agent and execute tools).


```python

agent_executor = AgentExecutor(agent=agent, tools=tools)  
  
question={"input": "give me articles written by Elyse Wanshel after 22nd of Sep about Comedy","chat_history":[],"agent_scratchpad":""}  
    
result=agent_executor.invoke(question)  
print("Answer", result['output'])
```

For the example we used: 
```json
{"input": "give me articles written by Elyse Wanshel after 22nd of Sep about Comedy"}
```

Answer:
I found an article written by Elyse Wanshel after September 22nd about Comedy:

- **Title:** 23 Of The Funniest Tweets About Cats And Dogs This Week (Sept. 17-23)
- **Category:** COMEDY
- **Short Description:** "Until you have a dog you don't understand what could be eaten."
- **Link:** [Read more](https://www.huffpost.com/entry/funniest-tweets-cats-dogs-september-17-23_n_632de332e4b0695c1d81dc02)



##### 6. Summary

In this blog post, we’ve implemented example for using   metadata filters using MongoDB, enhancing vector search accuracy and has minimal overhead compared to an unfiltered vector search.
There are other databases provide prefilter option for vector search like Neo4j, Weaviate and others.


You can take look of the full code from here 
