from huggingface_guess.detection import model_config_from_unet, unet_prefix_from_state_dict, model_config_from_diffusers_unet


def guess(state_dict):
    unet_key_prefix = unet_prefix_from_state_dict(state_dict)
    result = model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False)
    return result


def guess_repo_name(state_dict):
    config = guess(state_dict)
    assert config is not None
    repo_id = config.huggingface_repo
    return repo_id
