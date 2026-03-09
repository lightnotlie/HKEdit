from external.stylegan2.model import Generator
from utils.model_irse import IRSE
from torch import nn
import torchvision

import torch
from torchvision import utils
import os
import clip
import argparse

from train import TransModel

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str, required=True)
  parser.add_argument("--text", type=str, required=True, help='The input text for image editing')
  parser.add_argument("--gen_num", type=int, required=True, help='Number of generated samples')

  return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ckpt = 'pretrained/ffhq_256.pt'
    generator = Generator(256, 512, 8).cuda()
    generator.load_state_dict(torch.load(ckpt)['g_ema'], strict=False)
    generator.eval()

    model = TransModel(nhead=8, num_decoder_layers=6).cuda()
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict['state_dict'])
    model.clip_model = model.clip_model.float()
    model.eval()

    face_model = IRSE()
    face_model = nn.DataParallel(face_model).cuda()
    checkpoint = torch.load('pretrained/attribute_model.pth.tar')
    face_model.load_state_dict(checkpoint['state_dict'])
    face_model.eval()

    attributes = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
        'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
        'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
        'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    input_text = [args.text] * args.gen_num
    clip_text = clip.tokenize(input_text).cuda()

    truncation = 0.7
    truncation_mean = 4096
    save_path = 'generation'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    with torch.no_grad():
        mean_latent = generator.mean_latent(truncation_mean)

    code = torch.load('data/test_latents_seed100.pt')
    selected_idx = torch.randperm(len(code))[:args.gen_num]
    code = code[selected_idx].cuda()
    with torch.no_grad():
        styles = generator.style(code)
        input_im, _ = generator([styles], input_is_latent=True, randomize_noise=False, 
                        truncation=truncation, truncation_latent=mean_latent)

        offset = model(styles, clip_text)

        new_styles = styles.unsqueeze(1).repeat(1, 14, 1) + offset
        gen_im, _ = generator([new_styles], input_is_latent=True, randomize_noise=False, 
                        truncation=truncation, truncation_latent=mean_latent)

        in_attr = face_model(torchvision.transforms.functional.resize(input_im, 256))
        gen_attr = face_model(torchvision.transforms.functional.resize(gen_im, 256))

        in_preds = torch.stack(in_attr).transpose(0, 1).argmax(-1)
        gen_preds = torch.stack(gen_attr).transpose(0, 1).argmax(-1)

        for i in range(args.gen_num):

            input_attrs = []
            for j, attr_pred in enumerate(in_preds[i]):
                if attr_pred == 1:
                    input_attrs.append(attributes[j])
            output_attrs = []
            for j, attr_pred in enumerate(gen_preds[i]):
                if attr_pred == 1:
                    output_attrs.append(attributes[j])

        utils.save_image(input_im, save_path+"/input.png", nrow=args.gen_num, padding=10, normalize=True, range=(-1, 1), pad_value=1)
        utils.save_image(gen_im, save_path+"/output.png", nrow=args.gen_num, padding=10, normalize=True, range=(-1, 1), pad_value=1)

