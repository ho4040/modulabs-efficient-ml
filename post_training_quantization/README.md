# Post Training Quantization 

Pretrained 된 모델을 quantization 하는 경우입니다.  
오리지널 트레이닝 데이터가 필요하지 않으며, Calibration 과정을 통해서 Quantization 을 수행한다. Calibration 은 라벨링된 데이터 일 필요 없는, 새로운 데이터셋 만 있으면 가능합니다.

이 새로운 데이터셋을 모델에 입력으로 넣어 인퍼런스하는 과정에서 Statistics를 수집하여 각 레이어마다(혹은 세분성에 따라) scaling factor 와 zero point 를 찾도록 합니다.