import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from transformers import AutoImageProcessor, AutoModelForPreTraining, AutoModel,AutoFeatureExtractor, ViTMSNModel

list_of_string = ['facebook/vit-mae-base' ,
                   'facebook/vit-mae-huge',
                   'facebook/vit-mae-large' 
                  'microsoft/swin-base-simmim-window6-192',
                  'microsoft/beit-base-patch16-224',
                  'google/siglip-base-patch16-224',
                  'facebook/deit-small-distilled-patch16-224', #newly initialized: ['deit.pooler.dense.bias', 'deit.pooler.dense.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
                  'facebook/deit-small-patch16-224',        #newly initialized ['vit.pooler.dense.bias', 'vit.pooler.dense.weight'] You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
                  'openai/clip-vit-base-patch32',
                  'facebook/dinov2-small',

                  #erroneous models, some issues with transformers versioning
                  'facebook/dinov2-with-registers-small', 
                   'facebook/ijepa_vith14_1k',
                  ]

for s in list_of_string:
    processor = AutoImageProcessor.from_pretrained(s)
    model = AutoModel.from_pretrained(s)


