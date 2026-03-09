import clip
import numpy as np
import torch
import string

def produce_labels(sampled_text, anno, attr_list, gender_list, non_represents):
    clip_text = clip.tokenize(sampled_text)
    anno = np.array([int(a) > 0 for a in anno if a == '1' or a == '-1'], dtype = float)
    labels = anno.copy()
    all_labels = np.where(anno==1)[0]
    exist_mask = torch.zeros(40)
    color_related_mask = torch.zeros(40)
    structure_related_mask = torch.zeros(40)
    sampled_text = sampled_text.lower()
    sampled_text_nosign = ''.join([i for i in sampled_text if i not in string.punctuation])
    sampled_tokens = sampled_text_nosign.split(' ')
    for token in gender_list:
        if token in sampled_tokens:
            exist_mask[20] = 1
            break

    for i in range(len(all_labels)):
        attr_label = attr_list[all_labels[i]]
        if attr_label == 'male':
            continue                  
        if attr_label == 'no beard' and attr_label in sampled_text_nosign:
            exist_mask[all_labels[i]] = 1
            continue
        if attr_label == 'big nose' and attr_label in sampled_text_nosign:
            exist_mask[all_labels[i]] = 1
            continue
        tmp_exist = False
        split_attr_label = attr_label.split(' ')
        for a in split_attr_label:
            if a not in non_represents and a in sampled_text_nosign:
                tmp_exist = True
                exist_mask[all_labels[i]] = 1
                break
        if tmp_exist == False:
            labels[all_labels[i]] = 0
            if attr_label in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', "Pale_Skin"]:
                color_related_mask[all_labels[i]] = 1
            if attr_label in ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Big_Lips', 'Bald', 'Bangs', 'Bushy_Eyebrows',
                              'Double_Chin', 'Goatee', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes', 'Big_Nose',
                              'No_Beard', 'Receding_Hairline', 'Sideburns', 'Straight_Hair', 'Wavy_Hair',
                              'Pointy_Nose', 'Oval_Face', 'Chubby']:
                structure_related_mask[all_labels[i]] = 1
    return clip_text, labels, exist_mask, color_related_mask, structure_related_mask