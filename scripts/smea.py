try:
    import smea_dy
    from impl import sample_sa_solver, sample_sa_solver_dy, sample_sa_solver_pece, sample_sa_solver_pece_dy, sample_sa_solver_smea_dy, sample_sa_solver_pece_smea_dy
    from experimental import sample_sa_solver_renoise, sample_sa_solver_renoise_dy

    if smea_dy.BACKEND == "WebUI":
        from modules import scripts, sd_samplers_common, sd_samplers, script_callbacks, shared
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler
        import gradio as gr

        class SASmea(scripts.Script):
            def title(self):
                "SMEA Samplers"

            def show(self, is_img2img):
                return scripts.AlwaysVisible

            def __init__(self):
                if not smea_dy.INITIALIZED:
                    samplers_smea = [
                        ("SA-Solver", sample_sa_solver, ["k_sa_solver"], {}),
                        ("SA-Solver Dy", sample_sa_solver_dy, ["k_sa_solver_dy"], {}),
                        ("SA-Solver PECE", sample_sa_solver_pece, ["k_sa_solver_pece"], {}),
                        ("SA-Solver PECE Dy", sample_sa_solver_pece_dy, ["k_sa_solver_pece_dy"], {}),
                        ("SA-Solver SMEA Dy", sample_sa_solver_smea_dy, ["k_sa_solver_smea_dy"], {}),
                        ("SA-Solver PECE SMEA Dy", sample_sa_solver_pece_smea_dy, ["k_sa_solver_pece_smea_dy"], {}),
                        ("SA-Solver Experimental Renoise", sample_sa_solver_renoise, ["k_sa_solver_experimental"], {}),
                        ("SA-Solver Experimental Renoise Dy", sample_sa_solver_renoise_dy, ["k_sa_solver_experimental"], {}),
                        
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
        def on_ui_settings():
            section = ('SA-Solver', "SA-Solver")
            shared.opts.add_option("renoise_scale", shared.OptionInfo(
            default = 1.0,
            label = "Renoise Scale",
            component = gr.Slider,
            component_args = { 
                            'interactive': True, 
                            'minimum':0, 
                            'maximum':100, 
                            'step':0.01, 
                            },
            section = section
            ))
            shared.opts.add_option("renoise_seed", shared.OptionInfo(
            default = 1,
            label = "Renoise Seed",
            component = gr.Slider,
            component_args = { 
                            'interactive': True, 
                            'minimum':0, 
                            'maximum':0xffffffffffffffff, 
                            'step':1, 
                            },
            section = section
            ))
        script_callbacks.on_ui_settings(on_ui_settings)

except ImportError as _:
    print("Failed!")
    pass

from modules import devices, script_callbacks, shared
