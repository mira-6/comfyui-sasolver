import comfy
from functools import partial
from . import sa_solver

class SamplerSASolver:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "pc_mode": (['PEC', "PECE"],),
                    "smea": ("BOOLEAN", {"default": False}),
                    "dyn": ("BOOLEAN", {"default": False}),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    "eta_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "eta_end_percent": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "noise_device": (["cpu", "gpu"],),
                    }
                }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, model, pc_mode, smea, dyn, eta, eta_start_percent, eta_end_percent, noise_device):
        model_sampling = model.get_model_object('model_sampling')
        start_sigma = model_sampling.percent_to_sigma(eta_start_percent)
        end_sigma = model_sampling.percent_to_sigma(eta_end_percent)
        tau_func = partial(sa_solver.default_tau_func, eta=eta, eta_start_sigma=start_sigma, eta_end_sigma=end_sigma)
        
        if pc_mode == 'PEC':
            if smea == False and dyn == False:
                sampler_name = "sa_solver" if noise_device == "cpu" else "sa_solver_gpu"
            elif smea == True and dyn == False:
                raise ValueError("Dyn must be true to use smea!")
            elif smea == False and dyn == True:
                sampler_name = "sa_solver_dy" if noise_device == "cpu" else "sa_solver_dy_gpu"
            else:
                sampler_name = "sa_solver_smea_dy" if noise_device == "cpu" else "sa_solver_smea_dy_gpu"
        else:
            if smea == False and dyn == False:
                sampler_name = "sa_solver_pece" if noise_device == "cpu" else "sa_solver_pece_gpu"
            elif smea == True and dyn == False:
                raise ValueError("Dyn must be true to use smea!")
            elif smea == False and dyn == True:
                sampler_name = "sa_solver_pece_dy" if noise_device == "cpu" else "sa_solver_pece_dy_gpu"
            else:
                sampler_name = "sa_solver_pece_smea_dy" if noise_device == "cpu" else "sa_solver_pece_smea_dy_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"tau_func": tau_func})
        return (sampler, )
    
class SamplerSASolverExperimental:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "vae": ("VAE",),
                    "pc_mode": (['PEC', "PECE"],),
                    "invert": ("BOOLEAN", {"default": False}),
                    "normalize": ("BOOLEAN", {"default": False}),
                    "smea": ("BOOLEAN", {"default": False}),
                    "dyn": ("BOOLEAN", {"default": False}),
                    "renoise": ("BOOLEAN", {"default": False, "tooltip": "scale = 1.05 is recommended for renoise."}), # scale = 1.05 is recommended for renoise.
                    "renoise_alternative": ("BOOLEAN", {"default": False}),
                    "renoise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    "eta_start_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "eta_end_percent": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "scale": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": False}),
                    "shift": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01, "round": False}),
                    "gamma": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1, "round": False})
                    }
                }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    
    FUNCTION = "get_sampler"

    def get_sampler(self, model, vae, pc_mode, invert, normalize, smea, dyn, renoise, renoise_alternative, renoise_scale, eta, eta_start_percent, eta_end_percent, scale, shift, gamma):
        model_sampling = model.get_model_object('model_sampling')
        start_sigma = model_sampling.percent_to_sigma(eta_start_percent)
        end_sigma = model_sampling.percent_to_sigma(eta_end_percent)
        tau_func = partial(sa_solver.default_tau_func, eta=eta, eta_start_sigma=start_sigma, eta_end_sigma=end_sigma)
        sampler = comfy.samplers.ksampler("sa_solver_experimental", {"tau_func": tau_func, "vae": vae, "pc_mode": pc_mode, "invert": invert, "normalize": normalize, "smea": smea, "dyn": dyn, "scale": scale, "shift": shift, "gamma": gamma, "renoise": renoise, "renoise_alternative": renoise_alternative, "renoise_scale": renoise_scale})
        return (sampler, )