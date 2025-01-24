from transformers import AutoImageProcessor, AutoModelForPreTraining, AutoModel,AutoFeatureExtractor, ViTMSNModel
processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-base-4")
model = AutoModel.from_pretrained("facebook/vit-msn-base-4")
