import os
from torch.utils.data import DataLoader, TensorDataset

from utils.utils import load_data, get_args
from utils.train_loop import train_full_loop
from models.deep_model import DeepModel, DeepSAFEModel
from models.swin_transformer import SwinTransformer, SwinTransformerSAFE
 
from sklearn.model_selection import train_test_split  

from models.mambavision import MambaVisionForImageClassification, MambaVisionForImageClassification_v2
from models.nextvit import NextViTForImageClassification, NextViTForImageClassification_v2
from models.swintransformerv2 import SwinTransformerV2ForImageClassification, SwinTransformerV2ForImageClassification_v2
from models.convnextv2 import ConvNeXtV2ForImageClassification, ConvNeXtV2ForImageClassification_v2
from models.efficientnetv2 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v2
from models.gpt_vit import GPT2ForImageClassification
from models.bert_vit import BertForImageClassification,BertForImageClassification_v2
from models.efficientnetv2 import Hybrid_CNN_Transformer_AblationModel
import datetime
from zoneinfo import ZoneInfo  

os.makedirs('./model_save', exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('='*30)
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('='*30)
 
args = get_args()
class_name = args.class_name
print(f'now training for class: {class_name}')

X, y = load_data(class_name)

from sklearn.model_selection import KFold

kf = KFold(n_splits=3,shuffle=True,random_state=42)
kf.get_n_splits(X)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    train_idx=train_index[len(train_index)*0:int(len(train_index)*0.75)]
    validation_idx=train_index[int(len(train_index)*0.75):len(train_index)]

    X_train1,y_train1= X[train_idx],y[train_idx]
    X_val1,y_val1 = X[validation_idx],y[validation_idx]
    X_test1, y_test1=X[test_index],y[test_index]
    print(len(X_train1),len(X_val1),len(X_test1))

    train_dataset1 = TensorDataset(X_train1, y_train1)
    val_dataset1 = TensorDataset(X_val1, y_val1)
    test_dataset1 = TensorDataset(X_test1, y_test1)
 
    train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True)
    val_loader1 = DataLoader(val_dataset1, batch_size=32, shuffle=False)
    test_loader1 = DataLoader(test_dataset1, batch_size=32, shuffle=False)
    '''
    model_dicts = {
        'efficientnetv2_proposal': EfficientNetV2ForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        #'mambavision_proposal': MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'), 
        'mambavision': MambaVisionForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'),
        'nextvit': NextViTForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='small'),
        'efficientnetv2': EfficientNetV2ForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        'Resnet50': DeepModel('ResNet50'),
        'DenseNet121': DeepModel('DenseNet121'),
    }
    '''
    # 2. 모델 딕셔너리에 Ablation Study 모델 추가
    model_dicts = {
        # ---------------- Ablation Study Models ----------------
        # 1) Base: 제안 모듈 완전히 제외, 기존 백본(CNN) 특징만으로 분류
        'cnn_base_only': Hybrid_CNN_Transformer_AblationModel(
            num_labels=2, model_variant='s', hidden_dim=512,
            use_transformer=False, use_cross_attn=False
        ),

        # 2) w/o Cross-Attention: 트랜스포머는 통과하나, 교차 어텐션 없이 단순 합산(Addition)으로 융합
        'hybrid_wo_cross_attn': Hybrid_CNN_Transformer_AblationModel(
            num_labels=2, model_variant='s', hidden_dim=512,
            use_transformer=True, use_cross_attn=False
        ),

        # 3) Proposed (Full Model): CNN + Transformer + Cross-Attention 융합이 모두 적용된 최종 모델
        'hybrid_proposed': Hybrid_CNN_Transformer_AblationModel(
            num_labels=2, model_variant='s', hidden_dim=512,
            use_transformer=True, use_cross_attn=True
        ),
    }
    train_full_loop(train_loader1, val_loader1, test_loader1, model_dicts, class_name, i)

print('finished')
