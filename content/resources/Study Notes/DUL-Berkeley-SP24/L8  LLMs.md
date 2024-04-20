---
title: Large Language Models  (Lecture 8)
draft: false
tags:
  - NLP
  - LLM
  - Transformers
  - Study-Notes
  - DUL
date: <% tp.file.creation_date() %>
---

##  **Lecture 8: Large Language Models** (***LLMs***):
###  Deep Unsupervised Learning - Berkeley - Sp24 - L8:
- The Note is for  [Lecture 8](https://www.youtube.com/watch?v=tCgX48cvuw4) of [DUL Berkeley Spring 2024 offering](https://sites.google.com/view/berkeley-cs294-158-sp24/home), 
- Guest lecturer [Hao Liu](https://www.haoliu.site/), a Final Year PhD. student at Berkeley
- The Lecture Notes are work under construction, The lecture is much richer in information, and I'm working on the notes for the rest of the lecture. I'll update the content with the rest of the lecture ASAP.
---
### Language Model Abstraction:

- A Language Model is a probability distribution over sequences:
	-  Likelihood distribution: $p_{\theta}(y)$
	- Uses sequence of tokens $y ={( y_1 , ... , y_T)}$
	- Parameters ${\theta}$
	  
- Typically Language Models are autoregressive since it factorize sequences auto regressively.
	- Formula:		$$ p_{\theta} = \prod_{t=1}^{T}{p_{\theta}(y_{t}| y < t )}  $$
- Models are parameterized by a transformer.
  
-  So in summary the model is an autoregressive likelihood distribution over a sequence of tokens.
  
- Autoregressive Factorization, despite the other different ways to model language sequences, is done for:
	- Every Bit becomes part of the supervision, predicting every token and the tokens become prediction targets. 
		- ***Next Token Prediction***
		- It turned out to be crucial for the scaling of the language model.
	- Natural for Conversational AI.
		- Models suitable for following up conversation by ***Predicting Future Tokens***.
	
- Maximum Likelihood:
	-  The Goal is to make the observed data likely predictable under the model.
		- The **Objective Function** Formula: $$\underset{\theta}{\arg\max} \frac{1}{D} \sum_{y \in D} \log(p_{\theta}(y))$$
			- D is for example 1.5 Trillion Tokens in LLaMA.
			- ${\theta}$ is for example 7 to 70 billion parameters.
---
### Language Model Training:
 ####  **Stages of Learning**

1. Pretraining:
	- Done in Large scale.
	- Unsupervised Learning.
	- Maximizing Likelihood over a Very Large dataset of sequences.
   
2. Finetuning:
	- Model Specializing stage.
	- Focusing on specific set of tasks.
   
3. Learning from feedback:
	- Improving the model by giving feedback.
	- Focusing on outputs feedbacks such as human feedback in RLHF and debug messages or other methods like DPO.
	- The model is generally good and better on the fine-tuned task, but it needs to comprehend and follow user's intention so it needs users' feedback, and this stage focuses on how to use human feedback to align the model outputs with user desired outputs.

  
The Lecture focuses mostly on Pretraining LLMs and then will pass by some fine-tuning and learning from feedback.
- Because Most of the time, basically pretraining stage determines how good the model you built is.  

The reason behind choosing ***Unsupervised Learning*** :

- Because of the richness in unsupervised data (unlabeled data) sources.
- Better Generalization in most of the tasks as the model captures the distribution better.
	- it's shown better results than supervised learning in literature.
- For example, from the Llama paper the pretraining stage:
  
	![[Attachments/Llama-pretraining-graph.png]]
	
	- Y-axis: The Average most Likelihood Loss (Lower is better)
	- X-axis: Number of Tokens seen during training stage.
	- Notice that the curves aren't saturated and are continuously decreasing.
		- We still have way more available data on the internet for language learning.
---
### Scaling Compute

> **“The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.”**
> — _The Bitter Lesson, Richard Sutton, 2019_

- In summary: The major AI successes is going to come from methods that leverage computation
	- Typically the goal is to model the data distribution better by adding more compute.
	- Compute
		- It's the forward and backward pass (back-propagation).
		- Mostly spent on matrix multiplications (matmul)
		- Measured using (FLOPs)
			- As MatMuls are basically doing floating points multiplications and additions.
		- Compute Cost Estimating Formula: $$C = 6ND( 1+ \frac{s}{6d} )$$
			- N: Number of parameters, basically (***model***) size.
			- D: (***data***) size.
			- s: context size.
			- d: (***hidden***) dimension, the size projection dimension.
		- It turned out that size matters, lol.
		- TL;DR Using more tokens (larger dataset), larger context windows (longer sequences of data), and larger models add up compute costs.
			- Does that mean AGI will require the electricity of a country's households ?
---
### Tokens:

- Byte-Based Tokenization : 
	- Most Generic for all types of data sequences.
	- It's too long, leading to a lot more compute cost.
	- little bit unnecessary
	  
- Character-Based Tokenization :
	- Similar problem as Byte-Based long words require too many tokens.
	  
- Word-Based Tokenization :
	- Some words will share similar semantic meaning but different tokenization.
		- cat and cats should have the same semantic meaning. (Sorry, am a cat 😸 guy)
		  
- Sub-word-Based Tokenization :
	- A trade-off in between both worlds.
	- Byte-Pair Encoding: Replacing top appearing pairs with a new token.
		- It has the heart of frequency based feature extraction algorithm in classic classification problems.
	- Repeating until having a dictionary for tokens and a vocab size of tokens
	  
In 2017, people researches tried training LSTM with more compute on sentiment analysis, where it learns a sentiment neuron (+ve / -ve) after training to predict the next word on a large dataset of Amazon reviews.
- They basically did autoregressive next token prediction.
- The following graphs helped them deduce that:
  
	![[Attachments/LSTM-Compute.png]]
		
	- X-axis Observations: 
		- There are some neurons can be used to control the output of the LSTM.
	- Y-axis Observations:
		- Number of LSTM's output that're either (+ve) or (-ve).
	- By controlling the values of certain neurons, you can influence the LSTM's generative output to generate either (+ve) sentiment or (-ve) sentiment in reviews.
		- what a scary observation for product review batting 👾💀.
		  
- The model was able to comprehend human sentiment (+ve / -ve) by benefitting from more compute, that was a difficult language task back 7 years ago.
- Visual heatmap of sentiment neurons' values was generated on top of a document composed of 6 random highly contrasted IMDB reviews:

	![[Attachments/Visual-LSTM-Compute.png]]
		
	-  Red for (-ve) and Green for (+ve) that was an advancement back in 2017.

Then came a new proposed architecture called ***Attention,*** which scales much better than LSTMs.[[Transformers | Transformer models]] asymptotically outperformed LSTMs because of better use of context and specially when context window increases.
	![[Attachments/Attention-beating-LSTM.png]]
	
-  Red-Lines for LSTMs, while Blue ones beating the up LSTMs are Transformers.
- Left Figure shows per token test-loss:
	- Looking at a particular position over the 1000 tokens which model predicts that particular token better.
	- X-axis: the token index.
	- Y-axis: Loss corresponding to that particular token.
	 
- Right Figure shows per model size (number of parameters) test-loss:
	- Transformers measured parameters excluded embedding parameters.
	- The bigger the model, the bigger the gap between the 2 models performance.
		- Which's obviously because of the wiser usage of context using attention mechanism.
	- X-axis: Parameters number - model size.
	- Y-axis: Loss corresponding to specific model size
	- By scaling-up model size [[Transformers]] show pretty linear scaling, while LSTMs saturate early.
	   
- Attention helps keeping focus and information on past tokens without forgetting them.
	- LSTM you had to maintain certain hidden states, but in attention you can directly attend to any past token, the model doesn't have this information bottleneck.
	- You can easily increase model parameters by adding bigger MLP networks
		- which are large matrix multiplications so it scales well with modern GPUs and TPUs.
---
### Pretraining Objectives

We talked about full autoregressive prediction objective used by models like GPT, LLaMA.
However, there are other objectives that define the training of the model, for example:
- Masked Token Prediction used by: BERT, ELMO.
- Prefix autoregressive prediction used by: T5. 
  
	![[Attachments/Objectives-training-LLMS.png]]
- Masked token prediction is an effective objective for masked language models, as it helps them learn the full semantic meaning of a sequence.
	- Example: Embedding Model for searching retrieval.
- Prefix autoregressive predictions are an interesting alternative to masked token prediction for PLMs, but they are not yet widely used.
	- Most of Encoder-Decoder models they can be reformulated as autoregressive prediction but with a different attention mask like a prefix mask, the Encoder is the bidirectional attentive while the autoregressive part is the decoder.
- For visualization and clarification:
	![[Attachments/attention-masking-visual.png]]
	
	- X-axis and Y-axis are both the sequence of tokens.
	- PLM is the Non-Causal Decoder Middle graph.
	- Full Autoregressive widely used GPT model is the Causal Decoder first graph.
		- Different Objects but approximately similar amount of FLOPs computationally.
	- The Last one, which is the Encoder-Decoder [[Transformers]] tends to out to be the most scalable.
		-  Upstream (-ve) Log-perplexity : Vanilla Transformer outperforms other models.
		- Downstream accuracy : Vanilla Transformer outperforms other models.
		- An interesting question to explore is how the performance of the vanilla model compares to that of other derived models in terms of FLOP efficiency/costs.
---
### LLM-Compute Costs

- Hidden size of (MLP) is usually 4d because of expanding factor of MLP network.
- LLaMA (s << 6d), so approximately $C = 6ND = 6 * 7  billion * 2  trillion = 8.4 x 10^{22}FLOPs$
- Empirical performance of model has power-law relationship with each factor:
	- -log-log correlation between factors.
	-  $N_{opt}(C), D_{opt}(C) = \underset{N,D s.t. FLOPs(N,D)=C} {\arg\min} L(N,D)$
	- $N_{opt} \propto C^{a}, D_{opt} \propto C^{b}$
	- $a+b=1$ as $C=6ND$
	- which to choose a (more compute to parameters) or b(more compute to tokens):
		- OpenAI (2020) gives more to parameters.
		- DeepMind(2022) gives more compute to tokens.
		- Best practice: Train different small models of your desired architecture on different data sizes and fit the constant yourself.
			- pick the lowest test loss and draw the line fitting the lowest (fit the lowest loss).
			- Coefficients of model scaling and data scaling vary with the data distribution of the dataset itself.
			-   Wiser Compute allocation is always better than scaling-up:
				![[Attachments/chinchilla-wiser-compute.png]]
			
			- Chinchilla outperforms Megatron by allocating compute better.
			  
			  
			  
- However, [LLaMA-3](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) Using 15T tokens on 8B, 70B parameters model made the performance improvements clear and as announced the models weren't hitting saturation / convergence and 8B model great performance on various benchmarks, with Llam3-8B doing better than Llama2-70B in some cases.
- The focus on both optimizing compute and FLOPs while increasing quality of training Token Count, where 15T didn't even saturate a 8B parameters model, will be the focus of upcoming foundational models development and research.
---
## Resources:
- Lecture 8 video : [Lecture 8](https://www.youtube.com/watch?v=tCgX48cvuw4).
-  [DUL Berkeley Spring 2024 offering](https://sites.google.com/view/berkeley-cs294-158-sp24/home)
-  Screenshots from the [Lecture 8 PDF](https://drive.google.com/file/d/13YWiY4LLv_qshkSglpDh2VRnAbBHB6SX/view?usp=drive_link).
- [LLaMA 3](https://llama.meta.com/llama3/)