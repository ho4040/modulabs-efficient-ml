# Dynamic range for actvation quantization

![image](https://i.imgur.com/Tt5wc1e.png)

KL다이버전스를 이용해서 Activation 퀀타이제이션 위치를 계산 할 수 있다는 언급이 있어서 혹시 minimum 값이 존재하는 형태로 커브가 그려지는건가 싶어서 테스트를 해보았습니다. 

![image](https://i.imgur.com/nvWeSC3.png)

VGG16 3번 convolution 레이어를 테스트 했습니다만 미니멈 값은 없고, 예상대로 클리핑하는 영역이 적을 수록 KL다이버젼스도 줄어들었습니다. 
너무 KL다이버젼스가 높지 않은 수준으로 적당히 고른다는 의미로 받아들이면 될 듯 합니다.