from huggingface_guess.detection import model_config_from_unet, unet_prefix_from_state_dict, model_config_from_diffusers_unet


def guess_repo_name(state_dict):
    unet_key_prefix = unet_prefix_from_state_dict(state_dict)
    model_config = model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=True)
    repo_id = model_config.huggingface_repo
    return repo_id
