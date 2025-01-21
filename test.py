import torch
import torch.nn as nn

# 모델 출력 (logits)
outputs = torch.tensor([[2.0, 1.0, 0.1],  # 첫 번째 예측 (logits)
                        [0.5, 2.5, 0.1],  # 두 번째 예측 (logits)
                        [0.2, 0.3, 0.5]]) # 세 번째 예측 (logits)

# 정답 레이블 (단어 인덱스)
captions = torch.tensor([0, 1, 2])  # 정답 인덱스

# CrossEntropyLoss 정의
criterion = nn.CrossEntropyLoss()

# 손실 계산
loss = criterion(outputs, captions)
print("CrossEntropy Loss:", loss.item())
