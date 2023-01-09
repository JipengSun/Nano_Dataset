import matplotlib.pyplot as plt
import torch
import numpy as np
import imageio
import skimage

def show_tensor_image(image_tensor):
    # reverse_transforms = transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #     transforms.Lambda(lambda t: t * 255.),
    #     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    #     transforms.ToPILImage(),
    # ])

    # Take first image of batch
    # Draw all images inside a tensor (N,W,H,C)
    image_np = image_tensor.detach().cpu().numpy()

    N = image_np.shape[0]
    plt.figure(figsize=(15,15*N))
    plt.axis('off')
    for idx in range(N):
        plt.subplot(1, N, idx+1)
        plt.imshow((np.clip(image_np[idx,:,:,:],0,1)*255).astype(np.uint8))

def show_tensor_first_image(image_tensor):
    # reverse_transforms = transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #     transforms.Lambda(lambda t: t * 255.),
    #     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    #     transforms.ToPILImage(),
    # ])

    # Take first image of batch
    # Draw all images inside a tensor (N,C,W,H)

    image_np = image_tensor.detach().cpu().numpy()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow((image_np[0,:,:,:]*255).astype(np.uint8))



def show_output_tensor(out_tensor):
    img_out = reshape_train_to_image(out_tensor)
    #plt.imshow(img_out)
    show_tensor_image(img_out)
    
def reshape_train_to_image(train_sensor):

    return train_sensor.permute((0,2,3,1))

def reshape_image_to_train(image_sensor):
    return image_sensor.permute((0,3,1,2))


def display_results(input_tensor,gt_tensor,out_tensor):
    show_tensor_image(gt_tensor)
    show_tensor_image(input_tensor)
    show_output_tensor(out_tensor)


def load_np_psf(PSF_path,w,h):
    output_img = []
    for idx, channel in enumerate(["r", "g", "b"]):
        psf = imageio.imread(PSF_path+"psf_crop_{}.png".format(channel))
        output_img.append(psf)
    stack = np.array(output_img).astype(np.float32)
    stack = stack/255
    padx1 = int(np.ceil((w-stack.shape[1])/2))
    padx2 = int(np.floor((w-stack.shape[1])/2))
    pady1 = int(np.ceil((h-stack.shape[2])/2))
    pady2 = int(np.floor((h-stack.shape[2])/2))
    return np.pad(stack,((0,0),(padx1,padx2),(pady1,pady2)),'constant',constant_values = 0)


def single_img_train(img_path):
    img_np = skimage.io.imread(img_path).astype('float32')/255.
    #plt.imshow(img_np)
    img_tensor = torch.from_numpy(img_np)
    img_tensor = torch.unsqueeze(img_tensor,0)
    train_tensor = reshape_image_to_train(img_tensor)
    return train_tensor


def plot_rgb_tensor(out):
    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(out[0,0,...].detach().cpu().numpy())
    fig.add_subplot(1,3,2)
    plt.imshow(out[0,1,...].detach().cpu().numpy())
    fig.add_subplot(1,3,3)
    plt.imshow(out[0,2,...].detach().cpu().numpy())



def normalize_zero_one(A):
    batch, channel,height, width = A.shape
    AA = A.clone()
    AA = AA.reshape(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.reshape(batch, channel,height, width)
    return AA