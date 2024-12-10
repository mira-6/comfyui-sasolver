# needed to duplicate this here to solve an inconsistency
from importlib import import_module
BACKEND = None
if not BACKEND:
    try:
        _ = import_module("modules.sd_samplers_kdiffusion")
        sampling = import_module("k_diffusion.sampling")
        sample = import_module("ldm_patched.modules.sample")
        BACKEND = "WebUI"
    except ImportError as _:
        pass

if not BACKEND:
    try:
        sampling = import_module("comfy.k_diffusion.sampling")
        sample = import_module("comfy.sample")
        BACKEND = "ComfyUI"
    except ImportError as _:
        pass



from functools import partial
import torch
from torchvision.transforms import functional as tvf
from torchvision.transforms import ToTensor, ToPILImage
if BACKEND == "WebUI":
    import sa_solver
    import smea_dy
else:
    from . import sa_solver
    from . import smea_dy
from tqdm import trange
from PIL import Image
import numpy as np

# Tensor to PIL
# from comfyui-pixel, MIT License
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8),
        mode="RGB",
    )

# Modify from: https://github.com/scxue/SA-Solver
# MIT license
@torch.no_grad()
def sample_sa_solver(model, x, sigmas, vae=None, extra_args=None, callback=None, disable=False, predictor_order=3, corrector_order=4, pc_mode="PEC", tau_func=None, noise_sampler=None, smea=False, dyn=False, invert=False, normalize=False, gamma=1.0, scale=1.0, shift=0, renoise=False, renoise_alternative=False):
    
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    if tau_func is None:
        if smea_dy.BACKEND == "ComfyUI":
            model_sampling = model.inner_model.model_patcher.get_model_object('model_sampling')
            start_sigma = model_sampling.percent_to_sigma(0.2)
            end_sigma = model_sampling.percent_to_sigma(0.8)
        else:
            # this is NOT the same.
            start_sigma = sigmas[0]
            end_sigma = sigmas[-1]
        tau_func = partial(sa_solver.default_tau_func, eta=1.0, eta_start_sigma=start_sigma, eta_end_sigma=end_sigma)
    tau = tau_func
    noise_sampler = partial(sa_solver.device_noise_sampler, x=x, noise_device='cpu') if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    sigma_prev_list = []
    model_prev_list = []

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        if i == 0:
            # Init the initial values
            denoised = model(x, sigma * s_in, **extra_args)
            model_prev_list.append(denoised)
            sigma_prev_list.append(sigma)
            if normalize == True:
               denoised = torch.nn.functional.normalize(denoised)
        else:
            # Lower order final
            predictor_order_used = min(predictor_order, i, len(sigmas) - i - 1)
            corrector_order_used = min(corrector_order, i + 1, len(sigmas) - i + 1)

            tau_val = tau(sigma)
            noise = None if tau_val == 0 else noise_sampler()

            # Do SMEA/Dy
            sigma_hat = sigmas[i] * (gamma + 1)
            dt = sigmas[i + 1] - sigma_hat
            if dyn == True:
                if sigmas[i + 1] > 0:
                    if i // 2 == 1:
                        x = smea_dy.dy_sampling_step(x, model, dt, sigma_hat, **extra_args)
                    if i // 2 == 0 and smea: 
                        x = smea_dy.smea_sampling_step(x, model, dt, sigma_hat, **extra_args)
                

            # Predictor step
            x_p = sa_solver.adams_bashforth_update_few_steps(order=predictor_order_used, x=x, tau=tau_val,
                                                             model_prev_list=model_prev_list, sigma_prev_list=sigma_prev_list,
                                                             noise=noise, sigma=sigma)
            
            

            # Evaluation step
            denoised = model(x_p, sigma * s_in, **extra_args)
            if invert == True:
                if i // 2 == 1:
                    to_tensor = ToTensor()
                    to_pil_image = ToPILImage()
                    decoded = tensor2pil(vae.decode(denoised))
                    image = to_pil_image(tvf.invert(to_tensor(decoded)))
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    denoised = vae.encode(image).cuda()
            if renoise == True:
                if i % 2 == 0:
                    noised = sample.prepare_noise(denoised, i, None).cuda()
                    denoised = torch.lerp(denoised, noised, 1 / i)
            if renoise_alternative == True:
                noised = sample.prepare_noise(denoised, i, None).cuda()
                denoised = denoised + (1 / i) * noised
            denoised = scale * denoised + shift
            model_prev_list.append(denoised) 

            # Corrector step
            if corrector_order_used > 0:
                x = sa_solver.adams_moulton_update_few_steps(order=corrector_order_used, x=x, tau=tau_val,
                                                             model_prev_list=model_prev_list, sigma_prev_list=sigma_prev_list,
                                                             noise=noise, sigma=sigma)                                     
            else:
                x = x_p

            del noise, x_p
            
            # Evaluation step for PECE
            if corrector_order_used > 0 and pc_mode == 'PECE':
                del model_prev_list[-1]
                denoised = model(x, sigma * s_in, **extra_args)
                model_prev_list.append(denoised)

            sigma_prev_list.append(sigma)
            if len(model_prev_list) > max(predictor_order, corrector_order):
                del model_prev_list[0]
                del sigma_prev_list[0]

        if callback is not None:
            callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})

    if sigmas[-1] == 0:
        # Denoising step
        x = model_prev_list[-1]
    else:
        x = sa_solver.adams_bashforth_update_few_steps(order=1, x=x, tau=0,
                                                       model_prev_list=model_prev_list, sigma_prev_list=sigma_prev_list,
                                                       noise=0, sigma=sigmas[-1])
    return x

@torch.no_grad()
def sample_sa_solver_renoise(model, x, sigmas, vae=None, extra_args=None, callback=None, disable=False, predictor_order=3, corrector_order=4, pc_mode="PEC", tau_func=None, noise_sampler=None, smea=False, dyn=False, invert=False, normalize=False, gamma=1.0, scale=1.0, shift=0, renoise=False, renoise_alternative=False):
    return sample_sa_solver(model, x, sigmas, vae=None, extra_args=extra_args, callback=callback, disable=disable, predictor_order=predictor_order, corrector_order=corrector_order, pc_mode=pc_mode, tau_func=tau_func, noise_sampler=noise_sampler, smea=False, dyn=False, invert=False, normalize=False, gamma=1.0, scale=1.05, shift=0, renoise=True, renoise_alternative=False)

def sample_sa_solver_renoise_dy(model, x, sigmas, vae=None, extra_args=None, callback=None, disable=False, predictor_order=3, corrector_order=4, pc_mode="PEC", tau_func=None, noise_sampler=None, smea=False, dyn=False, invert=False, normalize=False, gamma=1.0, scale=1.0, shift=0, renoise=False, renoise_alternative=False):
    return sample_sa_solver(model, x, sigmas, vae=None, extra_args=extra_args, callback=callback, disable=disable, predictor_order=predictor_order, corrector_order=corrector_order, pc_mode=pc_mode, tau_func=tau_func, noise_sampler=noise_sampler, smea=False, dyn=True, invert=False, normalize=False, gamma=1.0, scale=1.05, shift=0, renoise=True, renoise_alternative=False)