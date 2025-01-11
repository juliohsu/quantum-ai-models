# quantum-ai-models

This is a block of ai models (ml/nn) self implement with PyTorch, by applying quantum-inspired optimization method, specifically focusing on attention mechanisms. (OBS: a method centralized by mimic the concept of quantum annealing to optimize hard attention mechanisms)

Where generally the tradicionals models focus on the soft attention, distributing focus all around the inputs with its varying weighs. In other hand, 
our goal is the hard attention focus on specific important inputs, resulting a interpretable and a possible more efficent model.

Objective: make the model training cheaper and interpretable.

What-to-do: apply quantum tunneling to escape local minima and find the global minimium in a complex energy landscape.

How-to-do:
1 - implement hard attention<br>
2 - incorporate quantum annealing<br>
3 - training loop adjustment<br>

Files:
 - "model.py" is a file for training the model that uses QAHAN
 - "noisys_data.py" is a file that generates kind of realistic world data, with noises and etc

 Source: https://arxiv.org/abs/2412.20930