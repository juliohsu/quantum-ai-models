# quantum-ai-models

This is a block of ai models (ml/nn) self implement with PyTorch, by applying quantum-inspired optimization method for training ml/neural networks, specifically focusing on attention mechanisms. (OBS: a method centralized by mimic the concept of quantum annealing to optimize hard attention mechanisms)

Where generally the tradicionals models focus on the soft attention, distributing focus all around the inputs with its varying weighs. In other hand, 
our goal is the hard attention focus on specific important inputs, resulting a interpretable and a possible more efficent model.

Objective: make the model training cheaper and interpretable.

Resume: apply quantum tunneling to escape local minima and find the global minimium in a complex energy landscape.

Code explain:<br>
1 - implement hard attention<br>
2 - incorporate quantum annealing<br>
3 - training loop adjustment<br>

How to run the code to test it out?<br>
Firstly, try to run the code with the model2 and the clean dataset, then change the dataset to the noisy one and run again.
From that aspect, you will notice that the model2 will break if we try to run it with the noisy dataset, so its time to show
our quantum model "model", and it will run normally without breaking out! Cool right?

Files:
 - "model.py" is a file of self implement PyTorch linear regression model that applies QAHAN techniques.
 - "noisys_data.py" is a file of dataset that generates kind of realistic world data, with noises and etc.

 Source: https://arxiv.org/abs/2412.20930