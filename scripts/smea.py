try:
    import smea_dy
    from impl import sample_sa_solver, sample_sa_solver_dy, sample_sa_solver_pece, sample_sa_solver_pece_dy, sample_sa_solver_smea_dy, sample_sa_solver_pece_smea_dy

    if smea_dy.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

        class SASmea(scripts.Script):
            def title(self):
                "SMEA Samplers"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def __init__(self):
                if not smea_dy.INITIALIZED:
                    samplers_smea = [
                        ("Sa-Solver", sample_sa_solver, ["k_sa_solver"], {}),
                        ("Sa-Solver Dy", sample_sa_solver_dy, ["k_sa_solver_dy"], {}),
                        ("Sa-Solver PECE", sample_sa_solver_pece, ["k_sa_solver_pece"], {}),
                        ("Sa-Solver PECE Dy", sample_sa_solver_pece_dy, ["k_sa_solver_pece_dy"], {}),
                        ("Sa-Solver SMEA Dy", sample_sa_solver_smea_dy, ["k_sa_solver_smea_dy"], {}),
                        ("Sa-Solver PECE SMEA Dy", sample_sa_solver_pece_smea_dy, ["k_sa_solver_pece_smea_dy"], {}),
                    ]
                    samplers_data_smea = [
                        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                        for label, funcname, aliases, options in samplers_smea
                        if callable(funcname)
                    ]
                    #sampler_extra_params["sample_euler_dy"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    #sampler_extra_params["sample_euler_smea_dy"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    #sampler_extra_params["sample_euler_negative"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    #sampler_extra_params["sample_euler_dy_negative"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    #sampler_extra_params["sample_Kohaku_LoNyu_Yog"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
                    sd_samplers.all_samplers.extend(samplers_data_smea)
                    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
                    sd_samplers.set_samplers()
                    smea_dy.INITIALIZED = True

except ImportError as _:
    print("Failed!")
    pass