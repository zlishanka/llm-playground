## LLM Distributed Model Training 

- The process of training a deep learning model across multiple devices or machines. 
- This approach is used to handle 
	- large datasets or 
	- complex models that may not fit into the memory of a single machine 

### Data Parallelism

- Dataset is divided into multiple subsets
- each subset is processed by a separate worker (e.g., CPU or GPU)
- Each worker computes the gradients for its subset of the data,
	- Each worker independently performs a "forward pass" through the model to compute predictions for its subset of the training data.
	- After computing the predictions, each worker computes the loss function (e.g., "cross-entropy loss") based on the ground truth 
		labels for its subset of the data.
	-  a backward pass is performed to compute the gradients of the loss function with respect to the model parameters,
		using techniques like backpropagation.
	- These local gradients represent the direction and magnitude of the change that would reduce the 
		loss for the data subset processed by each worker.
	- **Communication**, local gradients are computed, they need to be communicated to a `central node` or `parameter server` for aggregation.
	- **Aggregation**, At the central node or parameter server, the received gradients from all workers are aggregated to compute the `global gradient`.
		- `global gradient` represents the `average or sum of the gradients` computed by all workers across the entire training dataset.
		- may involve simple averaging of gradients or more sophisticated techniques like momentum or adaptive learning rate methods. 
	- **Update Model Parameters**, 
		- The updated parameters are then distributed back to the workers, and the process repeats for the next iteration of training. 
- and these "gradients are then aggregated" across all workers to update the model parameters. 

### Model Parallelism
	
- different parts of the model are executed on different devices.
- particularly useful when the model architecture is `too large to fit into the memory of a single device`. 
- Each device is responsible for "computing a portion of the model's forward and backward passes"
- communication is required to synchronize the updates across devices


### Challenges for Distributed Machine Learning

#### Synchronization	
- Synchronization is essential in distributed training to ensure that all workers are working on the "same version of the model parameters".
- This typically involves exchanging gradients or model updates between workers and applying these updates to the global model.

#### Communication
- Efficient communication is crucial in distributed training to minimize overhead and latency.
- Communication overhead can become a bottleneck, especially when training large models across many devices. 
- Techniques such as "asynchronous updates" or "gradient compression" can help reduce communication overhead.

#### Fault Tolerance
- Distributed training systems need to be resilient to failures
- Techniques such as "checkpointing"
- fault-tolerant communication protocols can help ensure that training can resume from a checkpoint 


### Hyperparameters
- "Batch size": number of training examples used in one iteration of gradient descent
	- static , typically 16 tokens, (Dynamic) GPT-3 increased from 32k to 3.2M
- "Learning rate": controls the step size during the optimization process.
	- learning rate increases linearly until reaching maximum value and then reduces vias a cosine decay until the learning rate is about 10% of its max value 

### Current frameworks that support distributed training 
	
- **TensorFlow with TensorFlow Distributed**, developed by Google [1]
	- `tf.distribute.MirroredStrategy` for multi-GPU training and `TPUStrategy` for TPU clusters
	- allows you to scale TensorFlow training across multiple devices, machines, or even across multiple data centers. 
	- It provides various strategies for distributed training, including data parallelism, model parallelism,

- **PyTorch with PyTorch Distributed**, developed by Facebook [5] 
	- - `DistributedDataParallel (DDP)` for data parallelism and `RPC` for model parallelism
	- enables you to scale PyTorch training across multiple devices and machines.
	- It offers various distributed training primitives and utilities to facilitate distributed training, 
		such as distributed data parallelism and distributed optimization algorithms.

- **Horovod**, developed by Uber
	- It supports TensorFlow, PyTorch, and other deep learning frameworks.
	- It leverages techniques like `ring-allreduce` for `efficient gradient aggregation` and supports common 
		distributed training patterns like data parallelism and model parallelism.	


## References
[1] Distributed training with Azure Machine Learning: https://learn.microsoft.com/en-us/azure/machine-learning/concept-distributed-training?view=azureml-api-2  
[2] Distributed Machine Learning Frameworks and its Benefits: https://www.xenonstack.com/blog/distributed-ml-framework  
[3] Distributed Training: Guide for Data Scientists: https://neptune.ai/blog/distributed-training  
[4] Parallel and Distributed Deep Learning: https://web.stanford.edu/~rezab/classes/cme323/S16/projects_reports/hedge_usmani.pdf  
[5] Distributed and Parallel Training Tutorials: https://docs.pytorch.org/tutorials/distributed/home.html  
