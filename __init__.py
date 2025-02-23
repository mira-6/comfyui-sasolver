from . import impl, smea_dy, experimental

if smea_dy.BACKEND == "ComfyUI":
    from comfy.k_diffusion import sampling as k_diffusion_sampling
    from comfy.samplers import SAMPLER_NAMES
    from . import nodes

    #regular
    setattr(k_diffusion_sampling, "sample_sa_solver", impl.sample_sa_solver)
    setattr(k_diffusion_sampling, "sample_sa_solver_gpu", impl.sample_sa_solver_gpu)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece", impl.sample_sa_solver_pece)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece_gpu", impl.sample_sa_solver_pece_gpu)

    #smea/dy
    setattr(k_diffusion_sampling, "sample_sa_solver_dy", impl.sample_sa_solver_dy)
    setattr(k_diffusion_sampling, "sample_sa_solver_dy_gpu", impl.sample_sa_solver_dy_gpu)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece_dy", impl.sample_sa_solver_pece_dy)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece_dy_gpu", impl.sample_sa_solver_pece_dy_gpu)

    setattr(k_diffusion_sampling, "sample_sa_solver_smea_dy", impl.sample_sa_solver_smea_dy)
    setattr(k_diffusion_sampling, "sample_sa_solver_smea_dy_gpu", impl.sample_sa_solver_smea_dy_gpu)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece_smea_dy", impl.sample_sa_solver_pece_smea_dy)
    setattr(k_diffusion_sampling, "sample_sa_solver_pece_smea_dy_gpu", impl.sample_sa_solver_pece_smea_dy_gpu)
    setattr(k_diffusion_sampling, "sample_sa_solver_experimental", experimental.sample_sa_solver)
    setattr(k_diffusion_sampling, "sample_sa_solver_experimental_renoise", experimental.sample_sa_solver_renoise)
    setattr(k_diffusion_sampling, "sample_sa_solver_experimental_renoise_dy", experimental.sample_sa_solver_renoise_dy)
    setattr(k_diffusion_sampling, "sample_sa_solver_experimental_renoise_a_dy", experimental.sample_sa_solver_renoise_a_dy)

    SAMPLER_NAMES.append("sa_solver")
    SAMPLER_NAMES.append("sa_solver_gpu")
    SAMPLER_NAMES.append("sa_solver_pece")
    SAMPLER_NAMES.append("sa_solver_pece_gpu")
    SAMPLER_NAMES.append("sa_solver_dy")
    SAMPLER_NAMES.append("sa_solver_dy_gpu")
    SAMPLER_NAMES.append("sa_solver_smea_dy")
    SAMPLER_NAMES.append("sa_solver_smea_dy_gpu")
    SAMPLER_NAMES.append("sa_solver_pece_dy")
    SAMPLER_NAMES.append("sa_solver_pece_dy_gpu")
    SAMPLER_NAMES.append("sa_solver_pece_smea_dy")
    SAMPLER_NAMES.append("sa_solver_pece_smea_dy_gpu")
    SAMPLER_NAMES.append("sa_solver_experimental")
    SAMPLER_NAMES.append("sa_solver_experimental_renoise")
    SAMPLER_NAMES.append("sa_solver_experimental_renoise_dy")
    SAMPLER_NAMES.append("sa_solver_experimental_renoise_a_dy")

    NODE_CLASS_MAPPINGS = {
        "SamplerSASolver": nodes.SamplerSASolver,
        "SamplerSASolverExperimental": nodes.SamplerSASolverExperimental
    }