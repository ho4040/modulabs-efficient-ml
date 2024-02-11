#%%
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#%%

# VGG16 모델 로드
model = models.vgg16(pretrained=True)
model.eval()  # 모델을 평가 모드로 설정
#%%
#%%
# 활성화 값을 저장할 변수
activation = {}

# 훅을 정의하여 특정 레이어의 출력을 캡처
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 3 번째 Convolutional layer에 훅 등록
model.features[5].register_forward_hook(get_activation('conv3'))
#%%
# 이미지 처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 이미지 로드 및 변환
img = Image.open("test.jpg")  # 이미지 경로를 지정하세요
img = transform(img).unsqueeze(0)  # 배치 차원 추가
#%%
# 모델에 이미지 전달
output = model(img)
#%%
# 활성화 값 추출
activations = activation['conv3'].squeeze().cpu().numpy()  # 배치 차원 제거 및 NumPy 배열로 변환
#%%
# 활성화 값의 분포 시각화
activations_flatten = activations.flatten()
activations_flatten_none_zero = activations_flatten[activations_flatten != 0] # 0 제외
counts, bins = np.histogram(activations_flatten_none_zero, bins=2000)
bins_centers = (bins[:-1] + bins[1:]) / 2
normalized_counts = counts / float(counts.sum())

#%%
def get_D_kl_value(samples, upper_bound):    
    epsilon = 1e-10  # 로그의 0 에러를 방지하기 위한 작은 값
    bins = np.linspace(0, 15, 100)
    counts, _ = np.histogram(samples, bins=bins)
    p = counts / float(counts.sum()) + epsilon
    x_cliped = np.clip(samples, 0, upper_bound)
    counts, _ = np.histogram(x_cliped, bins=bins)
    q = counts / float(counts.sum()) + epsilon
    D_KL = np.sum(p * np.log(p / q))
    return D_KL

clip_pos_list = []
d_kl_values = []
for i in range(1, 100, 1):
    clip = i/10
    v = get_D_kl_value(activations_flatten, clip)
    d_kl_values.append(v)
    clip_pos_list.append(clip)
#%%
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for activation values in Conv1 of VGG16
axs[0].scatter(bins_centers, normalized_counts, s=3, alpha=0.5)
axs[0].set_xlabel('Activation Value')
axs[0].set_ylabel('Normalized Number of Counts')
axs[0].set_yscale('log')
axs[0].set_title('Scatter Plot of Activation Values in Conv1 of VGG16')

# Plot for KL divergence by clip position
axs[1].plot(clip_pos_list, d_kl_values)
axs[1].set_xlabel('clip position')
axs[1].set_ylabel('KL divergence')
axs[1].set_title('KL divergence by clip position')

plt.tight_layout()
plt.show()
# %%
