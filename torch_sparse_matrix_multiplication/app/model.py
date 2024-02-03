import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.sparse import to_sparse_semi_structured


# MNIST 데이터셋을 위한 전처리 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 데이터셋 로드
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 기본 FC 모델 정의
class BasicFCModel(nn.Module):
    def __init__(self):
        super(BasicFCModel, self).__init__()
        self.fc1 = nn.Linear(784, 512).half() 
        self.hiddenLayer = nn.Linear(512, 512).half() # matrix shape: row=512, col=512. (행렬 사이즈가 32 또는 64의 배수인 경우만 작동합니다.)
        self.fc2 = nn.Linear(512, 10).half()
    def forward(self, x):
        x = x.view(-1, 784).half() # 입력 데이터를 784차원 벡터로 펼침
        x = F.relu(self.fc1(x))
        x = F.relu(self.hiddenLayer(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 프루닝 함수 정의
def apply_pruning(layer):
    with torch.no_grad():
        weight = layer.weight.data
        device = weight.device  # 현재 레이어의 디바이스
        shape = weight.shape
        row = shape[0]
        col = shape[1]
        mask = torch.ones_like(weight, device=device).bool() 
        # 각 4열마다 가장 작은 2개의 값을 0으로 설정
        for i in range(row):  
            for j in range(0, col, 4): 
                end = min(j + 4, col)  # 범위 초과 방지
                _, idx = torch.topk(weight[i, j:end].abs(), 2, largest=False)  # 가장 작은 2개 값의 인덱스
                mask[i, j:j+end][idx] = False  # 가중치가 0인 부분을 마스킹
        # 마스크 적용하여 가중치 수정
        layer.weight.data = layer.weight.data.masked_fill(~mask, 0)

def apply_compression(layer):
    layer.weight = nn.Parameter(to_sparse_semi_structured(layer.weight))


# 모델, 손실 함수, 최적화기 초기화
model = BasicFCModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 학습 함수
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 학습 시작
for epoch in range(1, 2):
    train(model, train_loader, optimizer, epoch)

# 학습 후 프루닝 적용
apply_pruning(model.hiddenLayer)

# 가중치 프루닝 확인
print("Hidden layer weights after pruning:", model.hiddenLayer.weight[0:4, 0:4])

# 압축된 행렬 확인
apply_compression(model.hiddenLayer)
print("Hidden layer weights after compression:", model.hiddenLayer.weight[0:4, 0:4])