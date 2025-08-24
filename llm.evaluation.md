## LLM Evaluation 

- Evaluating Language Model (LLM) performance involves assessing how well the model performs on specific tasks or datasets.
	 
### Core Performance Metrics

- **Perplexity**
	- Perplexity measures how well a probability distribution or probability model predicts a sample. [2]
	- Lower perplexity indicates better performance.
	- Commonly used to evaluate the language model's ability to predict the `next word` in a sequence.

- **Accuracy**
	- The proportion of correctly classified instances out of the total instances. [3] 
	- Measures the correctness of predictions in classification tasks.
	- Used in tasks like text classification, sentiment analysis, etc.

- **F1 score**
	- The harmonic mean of precision and recall.
	- Commonly used in binary and multi-class classification tasks.

### Generation Quality Metrics

- **BLEU Score** (Bilingual Evaluation Understudy Score)
	- Measures the quality of machine-translated text by comparing it to one or more reference translations. [2] [3]
	- Higher BLEU score indicates better translation quality.
	- Used in machine translation tasks to evaluate the quality of translations.

- **ROUGE Score** (Recall-Oriented Understudy for Gisting Evaluation)
	- Measures the quality of summaries by comparing them to one or more reference summaries.
	- Higher ROUGE score indicates better summary quality
	- Used in text summarization tasks to evaluate the quality of summaries.

- **Precision@K and Recall@K**
	- Precision@K measures the proportion of relevant instances among the top K predictions. 
	- Recall@K measures the proportion of relevant instances retrieved among all relevant instances.
	- Evaluates the performance of recommendation systems, search engines, etc., where only the top K predictions are considered.	
	- Used in ranking and recommendation tasks.

### Task-Specific and Human-Centric Metrics
- Human Evaluation
	- Involves human annotators assessing the quality of model outputs based on predefined criteria.
	- Provides insights into the model's performance from a human perspective.
	- Used when other automated metrics may not capture all aspects of model performance accurately.

- Cross-Validation
	- Divides the dataset into multiple subsets (folds) for training and evaluation.
	- Provides a more robust estimate of model performance by averaging results across multiple folds.
	- Used to evaluate model performance while minimizing the risk of overfitting.

- A/B Testing
	- Compares the performance of different models by exposing them to similar conditions and measuring their outcomes.
	- Determines which model variant performs better in real-world scenarios.
	- Used in production environments to assess the impact of model updates on user experience or business metrics.

- Domain-Specific Metrics
	- Metrics tailored to specific applications or domains, such as medical text analysis, legal document classification, etc.
	- Captures domain-specific nuances and requirements.
	- Used in specialized applications where general-purpose metrics may not be sufficient.

		
	
### Benchmark Datasets (open LLM leaderboard)
- Arc, HellaSwag, MMLU, TruthFulQA
- Multiple-choice Tasks, Arc, HellaSwag, MMLU
- Open-ended Tasks: TruthfulQA
	- Haman evaluation based ground truth, guideline
	- NLP metrics, quantify completion quality via metrics like Perplexity, BLEU or ROGUE scores
	- Auxiliary Fine-Tuned LLM, use LLM to compare completions to ground truth

#### ARC
- ARC is a dataset introduced by the Allen Institute for Artificial Intelligence (AI2) 
- for evaluating language models' ability to perform elementary science reasoning.
- contains questions that require various forms of reasoning, deductive, inductive, and abductive reasoning, 
	across a wide range of science topics. 
			
#### HellaSwag
- A benchmark dataset introduced by Salesforce Research. 
- It consists of examples that require commonsense reasoning beyond simple pattern matching or knowledge retrieval.	
- Aims to evaluate language models' ability to perform sophisticated commonsense reasoning tasks

#### MMLU (Minimal Manual Labeling Unit)
- A method for evaluating language models proposed by OpenAI. 
- It involves designing small, targeted tasks that can be evaluated with minimal manual labeling effort.
- Aims to assess language models' performance on specific linguistic phenomena or capabilities while 
	minimizing the annotation burden. 
- evaluate language models on fine-grained linguistic tasks, allowing researchers to gain insights into 
	the model's strengths and weaknesses in various linguistic domains.

#### TruthFulQA
- TruthfulQA is a benchmark dataset introduced by Google Research
- It consists of questions that require factually accurate answers, aiming to evaluate language models' 
	ability to provide trustworthy and reliable information.
- focuses on assessing language models' performance in providing accurate and reliable information, 
	particularly in the context of question answering.
- emphasizing the importance of factual accuracy in natural language understanding tasks.

## References
[1] LLM Evaluation: Key Metrics and Frameworks: https://aisera.com/blog/llm-evaluation/  
[2] LLM Evaluation: Metrics, Methodologies, Best Practices: https://www.datacamp.com/blog/llm-evaluation  
[3] Evaluating Large Language Models (LLMs): A Standard Set of Metrics for Accurate Assessment: https://www.linkedin.com/pulse/evaluating-large-language-models-llms-standard-set-metrics-biswas-ecjlc/  
