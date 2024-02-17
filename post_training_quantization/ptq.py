#%%
import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
# 랜덤시드 고정
_ = torch.manual_seed(0)
#%%
# MNIST 데이터셋을 불러와 사용할 수 있도록 준비
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
# 퀀타이제이션을 적용 할 기반 모델을 정의
class VerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

net = VerySimpleNet().to(device)
# %%
# 모델을 학습시키는 함수 정의
def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_iterations = 0
    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return
#%%
# 모델을 학습
train(train_loader, net, epochs=1)
#%%
# 모델의 크기를 확인하는 함수
def get_model_size(model):
    torch.save(model.state_dict(), "temp_delme.p") # 임시저장
    v = os.path.getsize("temp_delme.p")/1e3
    os.remove('temp_delme.p')
    return v

def print_size_of_model(model):
    print('모델크기(KB):', get_model_size(model))

#%%
# 모델의 Accuracy 확인 함수 정의
def test(model: nn.Module, total_iterations: int = None):
    correct = 0
    total = 0
    iterations = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total +=1
            iterations += 1
            if total_iterations is not None and iterations >= total_iterations:
                break
    acc = round(correct/total, 3)
    print(f'Accuracy: {acc}')
    return acc
#%%
# 퀀타이제이션 이전의 파라미터 값 확인
print('퀀타이제이션 이전의 파라미터 값')
print(net.linear1.weight)
print(net.linear1.weight.dtype)
print('퀀타이제이션 이전의 모델 사이즈')
print_size_of_model(net)
print(f'퀀타이제이션 이전의 정확도')
test(net)
# %%
# 퀀타이제이션을 적용할 모델 정의
class QuantizedVerySimpleNet(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(QuantizedVerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x
# %%
# 퀀타이제이션을 적용할 모델 생성
net_quantized = QuantizedVerySimpleNet().to(device)
net_quantized.load_state_dict(net.state_dict()) # 기존 모델을 퀀타이제이션 모델에 복사
net_quantized.eval()
net_quantized.qconfig = torch.ao.quantization.default_qconfig
net_quantized = torch.ao.quantization.prepare(net_quantized) # 퀀타이제이션용 레이어 준비 
net_quantized
# %%
# 테스트 배치를 통해서 activation 값을 관찰하여 min max 값을 계산함
test(net_quantized)
# %%
print(f'각 레이어마다 min_value와 max_value를 확인함')
net_quantized
# %%
# 퀀타이제이션을 적용. 이 과정에서 min max 값이 사용됨
# scale과 zero_point를 계산하여 가중치와 activation을 양자화함
net_quantized = torch.ao.quantization.convert(net_quantized)
print(f'scale 값, zero_point 값이 계산되어 가중치와 activation이 양자화됨')
net_quantized
# %%
# 퀀타이제이션 이후의 가중치 행렬 확인
# torch.int_repr : https://pytorch.org/docs/stable/generated/torch.Tensor.int_repr.html
print('퀀타이제이션 이후 가중치 행렬:')
print(torch.int_repr(net_quantized.linear1.weight()))
# %%
print('원래 가중치 확인: ')
print(net.linear1.weight)
#%%
print('')
print(f'디퀀타이즈 가중치 확인: ')
print(torch.dequantize(net_quantized.linear1.weight()))
print('')
#%%
print('Reconstruction loss: ')
torch.mean(torch.abs(net.linear1.weight - torch.dequantize(net_quantized.linear1.weight())))
#%%
# Weight 히스토그램 비교
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(net.linear1.weight.cpu().detach().numpy().flatten(), bins=300)
ax[1].hist(torch.int_repr(net_quantized.linear1.weight()).cpu().detach().numpy().flatten(), bins=300)
ax[0].set_title('Original')
ax[1].set_title('Quantized')
plt.title('Weight histogram comparison')
plt.legend()
plt.show()
# %%
# 모델 사이즈 비교
v1 = get_model_size(net)
v2 = get_model_size(net_quantized)
# bar chart of model size
fig, ax = plt.subplots()
ax.bar(['Original', 'Quantized'], [v1, v2])
ax.set_ylabel('Model size (KB)')
plt.title('Model size comparison')
plt.show()
# %%
# 정확도 비교
acc1 = test(net)
acc2 = test(net_quantized)
# bar chart of accuracy
fig, ax = plt.subplots()
ax.bar(['Original', 'Quantized'], [acc1, acc2])
ax.set_ylabel('Accuracy')
plt.title('Accuracy comparison')
plt.show()
# %%
