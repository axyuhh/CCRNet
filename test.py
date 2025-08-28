import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import time
import net as net
from function import adaptive_instance_normalization, coral

def test_transform(size, crop):
    transform_list = []
    transform_list.append(transforms.Resize(256))
    transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='./out',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
transform = net.adain(512)
vgg = net.vgg

decoder.eval()
transform.eval()
vgg.eval()
toot = './experiments/'
decoder.load_state_dict(torch.load(toot + 'decoder_iter_160000.pth'))
transform.load_state_dict(torch.load(toot + 'transform_iter_160000.pth'))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)
transform.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    for style_path in style_paths:
        content = content_tf(Image.open(str(content_path)).convert('RGB'))
        style = style_tf(Image.open(str(style_path)).convert('RGB'))
        if args.preserve_color:
            style = coral(style, content)
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            enc_layers = list(vgg.children())
            enc_1 = nn.Sequential(*enc_layers[:4])
            enc_2 = nn.Sequential(*enc_layers[4:11])
            enc_3 = nn.Sequential(*enc_layers[11:18])
            enc_4 = nn.Sequential(*enc_layers[18:31])

            style_feats = enc_4(enc_3(enc_2(enc_1(style))))
            content_feats = enc_4(enc_3(enc_2(enc_1(content))))

            t1 = transform(content_feats, style_feats)
            output = decoder(t1)

        output = output.cpu()
        output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
            content_path.stem, style_path.stem, args.save_ext)
        save_image(output, str(output_name))
