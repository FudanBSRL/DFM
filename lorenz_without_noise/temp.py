import torch

base_path = f'./'
pre_train_ex = f'rb_pre_without_model_ex4'
ksi = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}.pt')
Lambda = torch.load(f'{base_path}/ksis/ksi_{pre_train_ex}_lambda.pt')

print(ksi)