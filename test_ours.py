import os
import torch
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import glob

from PIL import Image
from sklearn.decomposition import PCA

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
from model.unet_attn import UNet2DConditionModel
from model.visual import mask_generation


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output", help="Path to the output directory")
parser.add_argument("--image_path", type=str, default="./bus.jpg", help="Path to the input image")
parser.add_argument("--prompt1", type=str, default="color the bus blue", help="First prompt for the model")
parser.add_argument("--prompt2", type=str, default="color the bus green", help="Second prompt for the model")
parser.add_argument("--object", type=str, default="bus", help="Object to be colored")
parser.add_argument("--width_ratio", type=float, default=0.5, help="width")
parser.add_argument("--height_ratio", type=float, default=0.5, help="height")
parser.add_argument("--train_sample_num", type=int, default=30, help="train sample number")
parser.add_argument("--test_num", type=int, default=10, help="test sample number")
args = parser.parse_args()


#load model
model_id = "/data/yangyuqi/models/instruct-pix2pix"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    custom_pipeline="./ip2p_ours.py",
)
pipe.to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)    
pipe.unet = UNet2DConditionModel.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    subfolder="unet",
).to("cuda")


output_dir = args.output_dir
embedding_dir = "/data/yangyuqi/ip2p-result/bus/blue_green"
os.makedirs(output_dir, exist_ok=True)
#load image
image_path = args.image_path
init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((512, 512))  

#插值生成一组图片 theta
train_sample_num = args.train_sample_num
test_num = args.test_num
theta = np.linspace(0, 1 , num=train_sample_num)

height = int(init_image.size[1] * args.height_ratio)
width = int(init_image.size[0] * args.width_ratio)

prompt1 = args.prompt1
prompt2 = args.prompt2
object = args.object

mask = mask_generation(init_image, object, output_dir)

def save_embedding():
    imgs = []
    for theta_item in theta:

        img = pipe(prompt1, prompt2, object, mask, theta_item, path=None, mode='train', output_path=output_dir, image=init_image).images[0]
        img.save(output_dir + f'/{theta_item:.2f}.png')

        imgs.append(img) 

    plt.figure(figsize=(16,12))
    plt.subplot(4, 8, 1)
    plt.imshow(init_image)
    plt.title(f'Original Image', fontsize=10)
    plt.axis('off')

    for i, (alpha, img) in enumerate(zip(theta, imgs)):
        plt.subplot(4, 8 , i + 2)
        plt.imshow(img)
        plt.title(f'alpha = {alpha:.2f}', fontsize=10)
        plt.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_dir + '/output_inter.png')
    plt.show()


def load_embedding(theta):
    embedding_list = []
    embedding_path_list = [f'embedding_{i:.2f}.pt' for i in theta]
    for file_name in embedding_path_list:
        embedding_path = os.path.join(embedding_dir, file_name)
        embedding = torch.load(embedding_path)

        embedding_list.append(embedding['embedding'][0])  # shape: (77, 768)
    embedding_batch = torch.stack(embedding_list, dim=0)  # shape: (batch_size, 77, 768)

    return embedding_batch  # shape: (20, 77, 768)


def load_image_rgb(theta, width, height):
    image_rgb_list = []
    image_path_list = [f'{i:.2f}.png' for i in theta]
    for file_name in image_path_list:
        image_path = os.path.join(embedding_dir, file_name)
        image = Image.open(image_path)
        image = image.resize((512, 512))
        r, g ,b = image.getpixel((width, height))
        image_rgb = torch.tensor([r,g,b])

        image_rgb_list.append(image_rgb)  # shape: (3,)
    first_pixel = image_rgb_list[4]
    last_pixel = image_rgb_list[12]
    image_rgb_batch = torch.stack(image_rgb_list, dim=0)  # shape: (20, 3)

    return image_rgb_batch, first_pixel, last_pixel  # shape: (20, 3)

class Vec3ToPCAEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

    def forward(self, x):  # x: [batch, 3]
        return self.model(x)

def predict_embedding(embedding_batch, image_rgb_batch, first_pixel, last_pixel, n_components):

    batch_size = embedding_batch.shape[0]
    embedding = embedding_batch.cpu().reshape(batch_size, -1)  # [100, 77, 768]
    image_rgb = image_rgb_batch.cpu().reshape(batch_size, -1)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(embedding)  # [100, 512]

    model = Vec3ToPCAEmbedding()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_target = torch.tensor(X_pca, dtype=torch.float32)  # [100, 512]
    Y_input = torch.tensor(image_rgb, dtype=torch.float32)  # [100, 3]
    
    for epoch in range(500):
        pred = model(Y_input)  # [100, 512]
        loss = criterion(pred, X_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    pred_embeddings = []
    pixel_rgb = []
    with torch.no_grad():
        for i in range(test_num):
            alpha = i / (test_num-1)
            current_pixel = (1 - alpha) * first_pixel + alpha * last_pixel
            Y_input = torch.tensor(current_pixel,dtype=torch.float32)
            # Y_input = torch.tensor([0, 140 + i * 4 ,200 - 5 * i],dtype=torch.float32)

            pixel_str = '_'.join(str(int(x.item())) for x in current_pixel)
            pred_pca = model(Y_input).numpy()                # [100, 512]
            pred_flat = pca.inverse_transform(pred_pca)      # [100, 59136]
            pred_embedding = pred_flat.reshape(77, 768)         # [100, 77, 768]
            pred_embeddings.append((pred_embedding, pixel_str))  # [100, 77, 768]
            torch.save({'pred_embedding':pred_embedding}, output_dir + f'/pred_embedding_{i}_' + pixel_str + '.pt' )

    return pred_embeddings

def generate_image(pred_embeddings):

    imgs_output = []
    pixel_values = []
    i = 0
    for (pred_embedding, pixel_str) in pred_embeddings:
        img = pipe(prompt1, prompt2, object, mask, theta=0, pred_embedding=pred_embedding, mode='test',output_path=None, image=init_image).images[0]
        img.save(output_dir + f'/new_{i}_{pixel_str}.png')
        i += 1

        imgs_output.append(img)
        pixel_values.append(pixel_str)

    plt.figure(figsize=(16,14))
    plt.subplot(7, 8, 1)
    plt.imshow(init_image)
    plt.title(f'Original Image', fontsize=10)
    plt.axis('off')

    for i, img in enumerate(imgs_output):
        plt.subplot(7, 8 , i + 2)
        plt.imshow(img)
        plt.title(f'{i}_{pixel_values[i]}', fontsize=10)
        plt.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_dir + '/output_ours.png')
    plt.show()

def generate_image(pred_embeddings):

    imgs_output = []
    pixel_values = []
    for i in range(test_num):
        pattern = os.path.join(output_dir, f'pred_embedding_{i}*.pt')
        paths = glob.glob(pattern)
        if len(paths) == 0:
            print(f"[Warning] No file found for: {pattern}")
            continue
        path = paths[0] 
        filename = os.path.basename(path)
        parts = filename.replace(".pt", "").split("_")
        result = parts[-3:]

        img = pipe(prompt1, prompt2, object, mask, theta=0, path=path, mode='test',output_path=None, image=init_image).images[0]
        img.save(output_dir + f'/new_{i}_{result}.png')

        imgs_output.append(img)
        pixel_values.append(result)

    plt.figure(figsize=(16,14))
    plt.subplot(7, 8, 1)
    plt.imshow(init_image)
    plt.title(f'Original Image', fontsize=10)
    plt.axis('off')

    for i, img in enumerate(imgs_output):
        plt.subplot(7, 8 , i + 2)
        plt.imshow(img)
        plt.title(f'{i}_{pixel_values[i]}', fontsize=10)
        plt.axis('off')

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.subplots_adjust(top=0.9)
    plt.savefig(output_dir + '/output_ours.png')
    plt.show()


def main():

    save_embedding()
    embedding_batch = load_embedding(theta)
    image_rgb_batch, first_pixel, last_pixel = load_image_rgb(theta, width, height)  # 256, 341取RGB值的像素点坐标
    predict_embeddings = predict_embedding(embedding_batch, image_rgb_batch,  first_pixel, last_pixel, n_components=15)
    generate_image(predict_embeddings)

if __name__ == "__main__":  
    main()