#%%
import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# %%
_ = torch.manual_seed(0)
# %%
# MNIST dataset 준비
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
# %%
# 간단한 테스트용 모델 생성
class VerySimpleNet(nn.Module):
    # QAT 의 경우, 학습중에 Quantized activation 이 적용 된 상태로 학습을 진행하기 위해
    # 학습전에 weight_fake_quant 레이어가 적용된 모델을 만든다.

    def __init__(self, hidden_size_1=100, hidden_size_2=100):
        super(VerySimpleNet,self).__init__()
        self.quant = torch.quantization.QuantStub() # 이후 부분에 양자화 된 값을 전달하기 위한 모듈 
        self.linear1 = nn.Linear(28*28, hidden_size_1) 
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2) 
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub() # 양자화 된 값을 다시 원래 값으로 변환하기 위한 모듈
    
    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.quant(x) # 입력을 퀀타이제이션으로 한번 감싼다.
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.dequant(x)
        return x

net = VerySimpleNet().to(device)
net
# %%
# prepare_qat 함수를 이용하여 각 Linear 레이어마다 Quantization 하도록 만듭니다.
net.qconfig = torch.ao.quantization.default_qconfig
net.train()
# torch.ao.quantization.prepare_qat 레이어마다 weight_fake_quant 모듈을 붙여줍니다.
net_quantized = torch.ao.quantization.prepare_qat(net) 
net_quantized
# %%
# 학습 진행
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
# 모델의 크기를 확인용 함수 
def get_model_size(model):
    torch.save(model.state_dict(), "temp_delme.p")
    v = os.path.getsize("temp_delme.p")/1e3
    os.remove('temp_delme.p')
    return v

def print_size_of_model(model):
    print('Size (KB):', get_model_size(model))

#%%
# 학습을 진행합니다.
train(train_loader, net_quantized, epochs=1)
# %%
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

print(f'칼리브레이션 정보 확인')
net_quantized
# %%
# 수집된 칼리브레이션 정보를 이용해서 양자화를 완료합니다. 
net_quantized.eval()
net_converted = torch.ao.quantization.convert(net_quantized)
# %%
print(f'양자화 된 모델 정보 확인')
net_quantized
#%%
print(f'양자화 된 모델 정보 확인')
net_converted
# %%
print('퀀타이제이션 된 모델 가중치 확인')
print(torch.int_repr(net_converted.linear1.weight()))
#%%
print('Reconstruction loss: ')
torch.mean(torch.abs(net_quantized.linear1.weight - torch.dequantize(net_converted.linear1.weight())))
#%%
# Weight 히스토그램 비교
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(net_quantized.linear1.weight.cpu().detach().numpy().flatten(), bins=300)
ax[1].hist(torch.int_repr(net_converted.linear1.weight()).cpu().detach().numpy().flatten(), bins=300)
ax[0].set_title('After QAT')
ax[1].set_title('Apply Quantization')
plt.legend()
plt.show()

# %%
# 모델 사이즈 비교
v1 = get_model_size(net_quantized)
v2 = get_model_size(net_converted)
# bar chart of model size
fig, ax = plt.subplots()
ax.bar(['After QAT', 'Quantized'], [v1, v2])
ax.set_ylabel('Model size (KB)')
plt.title('Model size comparison')
plt.show()
# %%
# 정확도 비교
acc1 = test(net_quantized)
acc2 = test(net_converted)
# bar chart of accuracy
fig, ax = plt.subplots()
ax.bar(['After QAT', 'Quantized'], [acc1, acc2])
ax.set_ylabel('Accuracy')
plt.title('Accuracy comparison')
plt.show()
# %%
