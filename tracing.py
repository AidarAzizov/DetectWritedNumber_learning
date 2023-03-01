import torch
from MNISTNet import MNISTNet
from torchvision import transforms
from PIL import Image

input_path = 'inputs/'
output_path = 'outputs/'
image_path = input_path + 'mnist_formatted_7.png'
model_filename = output_path + 'model.pt'
traced_model_filename = output_path + 'traced_model.pt'


transform = transforms.Compose([
    transforms.ToTensor()
])


if __name__ == '__main__':

    print('loading model from', model_filename, '...')
    model = torch.load(model_filename)
    model.eval()
    print('loaded model')


    print('loading image from', image_path, '...')
    test_image = Image.open(image_path)
    test_image = transform(test_image)
    test_image *= 255.
    print('loaded image')


    out = model(test_image.unsqueeze(0))
    print('Checking model with', image_path, 'image...')
    print('Model predicted', out.argmax().item())


    traced_model = torch.jit.trace(model, test_image.unsqueeze(0))
    traced_model.save(traced_model_filename)
    print('Traced model saved to', traced_model_filename)


    loaded_trace_model = torch.jit.load(traced_model_filename)
    out_from_traced = loaded_trace_model(test_image.unsqueeze(0))
    print('Checking traced model with', image_path, 'image...')
    print('Model predicted', out_from_traced.argmax().item())
