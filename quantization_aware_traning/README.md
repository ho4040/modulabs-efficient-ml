# Quantization Aware Training

퀀타이제이션의 또 다른 전략입니다.
PTQ 가 이미 Pretrained 된 모델을 Quantization 하는 방법이었다면 QAT 는 Quantization 에 좀 더 잘 견디도록 모델을 학습하는 방법입니다. 

PTQ가 학습이 끝나고 Quantization 하는 것과 달리 QAT 학습을 도중에 Quantization 을 진행합니다.
모델에 Quantization 이 된 것 같은 효과를 주는 Fake quantization 모듈을 포함하여 학습 합니다.
결과적으로 Loss 에 Quantization 에 의한 오류가 포함됩니다. 
이 덕분에 Weight 는 실제로 Quantization 이 되더라도 Accuracy 의 손실을 최소화 할 수 있습니다.