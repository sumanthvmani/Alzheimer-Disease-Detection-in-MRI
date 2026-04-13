import numpy as np
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

No_of_Dataset = 2

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()

for n in range(No_of_Dataset):
    Images = np.load('Images_' + str(n+1) + '.npy')[:5]

    for i in range(len(Images)):

        img = Images[i]                        # numpy array (H,W,3)
        img_pil = Image.fromarray(img.astype(np.uint8))  # FIX ✔

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        input_tensor = transform(img_pil).unsqueeze(0)

        finalconv_name = 'layer4'
        features, gradients = [], []

        def save_features(m, i, o):
            features.append(o)

        def save_gradients(m, gi, go):
            gradients.append(go[0])

        layer = dict([*model.named_modules()])[finalconv_name]
        layer.register_forward_hook(save_features)
        layer.register_backward_hook(save_gradients)

        output = model(input_tensor)
        pred_class = output.argmax().item()

        model.zero_grad()
        output[0, pred_class].backward()

        grads = gradients[0].cpu().detach().numpy()[0]
        fmap = features[0].cpu().detach().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))

        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for j, w in enumerate(weights):
            cam += w * fmap[j]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (0.4 * heatmap + 0.6 * img).astype(np.uint8)
        cv2.imwrite(f"./Results/Image_Results/Dataset_{n + 1}_Img_{i + 1}_org.jpg", img)
        cv2.imwrite(f"./Results/Image_Results/Dataset_{n+1}_Img_{i+1}_GradCAM.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
