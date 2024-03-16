---
title: General Plan
draft: false
tags:
  - NLP
  - project
  - recommender
  - content-based-filtering
date: <% tp.file.creation_date() %>
---

For my use case, I would recommend using **LlamaIndex**. Here's why:

1. **Semantic Similarity**: LlamaIndex is designed to work with Language Models, which are excellent at understanding semantic similarity. You can use it to index your anime descriptions and then retrieve similar animes based on the semantic similarity of their descriptions.

2. **Categorical Features**: While LlamaIndex primarily deals with text data, you can also use categorical features in your recommendation system. You can create a hybrid recommendation system that uses both the semantic similarity of the descriptions and the categorical features of the animes.

3. **Data Format**: LlamaIndex provides data connectors that can ingest data from various sources. So, whether your data is in CSV or Parquet format should not be a problem.

4. **Integration with Streamlit**: LlamaIndex can be used in conjunction with Streamlit to build a search bar interface. You can use Streamlit to take user input, pass it to your LlamaIndex-based recommendation system, and then display the top K similar animes.

Remember, while LlamaIndex can help you build the recommendation system, I'll also need a language model to compute the semantic similarity of the anime descriptions. 
I might want to consider models like BERT or GPT-3 for this purpose.😊
