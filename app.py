import gradio as gr

from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer


class App:
    def __init__(self,
                 args=None):
        self.args = args
        self.app = gr.Blocks()
        self.inferencer = LivePortraitInferencer()

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    img_ref = gr.Image(label="Reference Image")

                with gr.Column():
                    sld_rotate_pitch = gr.Slider(label="Rotate Pitch", minimum=-20, maximum=20, step=0.5, value=0)
                    sld_rotate_yaw = gr.Slider(label="Rotate Yaw", minimum=-20, maximum=20, step=0.5, value=0)
                    sld_rotate_roll = gr.Slider(label="Rotate Roll", minimum=-20, maximum=20, step=0.5, value=0)

                    sld_blink = gr.Slider(label="Blink", minimum=-20, maximum=5, step=0.5, value=0)
                    sld_eyebrow = gr.Slider(label="Eyebrow", minimum=-10, maximum=15, step=0.5, value=0)
                    sld_wink = gr.Slider(label="Wink", minimum=0, maximum=25, step=0.5, value=0)
                    sld_pupil_x = gr.Slider(label="Pupil X", minimum=-15, maximum=15, step=0.5, value=0)
                    sld_pupil_y = gr.Slider(label="Pupil Y", minimum=-15, maximum=15, step=0.5, value=0)

                    sld_aaa = gr.Slider(label="AAA", minimum=-30, maximum=120, step=1, value=0)
                    sld_eee = gr.Slider(label="EEE", minimum=-20, maximum=15, step=0.2, value=0)
                    sld_woo = gr.Slider(label="WOO", minimum=-20, maximum=15, step=0.2, value=0)
                    sld_smile = gr.Slider(label="Smile", minimum=-0.3, maximum=1.3, step=0.01, value=0)

                    sld_src_ratio = gr.Slider(label="Source Ratio", minimum=0, maximum=1, step=0.01, value=1)
                    sld_sample_ratio = gr.Slider(label="Sample Ratio", minimum=-0.2, maximum=1.2, step=0.01, value=1)

                    dd_sample_parts = gr.Dropdown(label="Sample Parts",
                                                  choices=["OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"],
                                                  value="All")
                    sld_crop_factor = gr.Slider(label="Crop Factor", minimum=1.5, maximum=2.5, step=0.1,
                                                value=1.7)
                    with gr.Accordion("Opt in features", visible=False):
                        img_sample = gr.Image()
                        img_motion_link = gr.Image()
                        tb_exp = gr.Textbox()
            with gr.Row():
                btn_gen = gr.Button("GEN")
            with gr.Row():
                img_out = gr.Image(label="Output Image")

            params = [
                sld_rotate_pitch, sld_rotate_yaw, sld_rotate_roll, sld_blink, sld_eyebrow, sld_wink, sld_pupil_x,
                sld_pupil_y, sld_aaa, sld_eee, sld_woo, sld_smile, sld_src_ratio, sld_sample_ratio, dd_sample_parts,
                sld_crop_factor, img_ref
            ]
            opt_in_features_params = [img_sample, img_motion_link, tb_exp]

            btn_gen.click(self.inferencer.edit_expression,
                          inputs=params + opt_in_features_params,
                          outputs=img_out)

        self.app.launch(inbrowser=True)


app = App()
app.launch()




