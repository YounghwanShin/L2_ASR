import torch

EDIT_SYMBOLS = {
    "eq": "=",
    "ins": "I",
    "del": "D", 
    "sub": "S",
}

def make_attn_mask(wavs, wav_lens):
    abs_lens = (wav_lens * wavs.shape[1]).long()
    attn_mask = wavs.new(wavs.shape).zero_().long()
    for i in range(len(abs_lens)):
        attn_mask[i, :abs_lens[i]] = 1
    return attn_mask

def get_model_class(model_type):
    if model_type == 'simple':
        from models.multitask_model import SimpleMultiTaskModel
        from models.loss_functions import MultiTaskLoss
        return SimpleMultiTaskModel, MultiTaskLoss
    elif model_type == 'transformer':
        from models.multitask_model_transformer import TransformerMultiTaskModel
        from models.loss_functions import MultiTaskLoss
        return TransformerMultiTaskModel, MultiTaskLoss
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: simple, transformer")

def get_phoneme_model_class(model_type):
    if model_type == 'simple':
        from models.phoneme_model import SimplePhonemeModel
        from models.loss_functions import PhonemeLoss
        return SimplePhonemeModel, PhonemeLoss
    elif model_type == 'transformer':
        from models.phoneme_model_transformer import TransformerPhonemeModel
        from models.loss_functions import PhonemeLoss
        return TransformerPhonemeModel, PhonemeLoss
    else:
        raise ValueError(f"Unknown phoneme model type: {model_type}. Available: simple, transformer")

def detect_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'
    else:
        return 'simple'

def detect_phoneme_model_type_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    state_dict = remove_module_prefix(state_dict)
    keys = list(state_dict.keys())
    
    if any('transformer_encoder' in key for key in keys):
        return 'transformer'
    elif any('shared_encoder' in key for key in keys):
        return 'simple'
