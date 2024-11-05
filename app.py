import gradio as gr

from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer


class App:
    def __init__(self,
                 args=None):
        self.args = args
        self.app = gr.Blocks()
        self.inferencer = LivePortraitInferencer()

    @staticmethod
    def create_parameters():
        return [
            gr.Slider(label="Rotate Pitch", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Rotate Yaw", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Rotate Roll", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Blink", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Eyebrow", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Wink", minimum=0, maximum=25, step=0.5, value=0),
            gr.Slider(label="Pupil X", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="Pupil Y", minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label="AAA", minimum=-30, maximum=120, step=1, value=0),
            gr.Slider(label="EEE", minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label="WOO", minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label="Smile", minimum=-0.3, maximum=1.3, step=0.01, value=0),
            gr.Slider(label="Source Ratio", minimum=0, maximum=1, step=0.01, value=1),
            gr.Slider(label="Sample Ratio", minimum=-0.2, maximum=1.2, step=0.01, value=1),
            gr.Dropdown(label="Sample Parts",
                        choices=["OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"], value="All"),
            gr.Slider(label="Crop Factor", minimum=1.5, maximum=2.5, step=0.1, value=1.7)
        ]

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    img_ref = gr.Image(label="Reference Image")
            with gr.Row():
                btn_gen = gr.Button("GENERATE", visible=False)
            with gr.Row():
                with gr.Column(scale=8):
                    img_out = gr.Image(label="Output Image")
                with gr.Column(scale=2):
                    expression_parameters = self.create_parameters()
                    with gr.Accordion("Opt in features", visible=False):
                        img_sample = gr.Image()
                        img_motion_link = gr.Image()
                        tb_exp = gr.Textbox()

            params = expression_parameters + [img_ref]
            opt_in_features_params = [img_sample, img_motion_link, tb_exp]

            changes_expressions = [param.change for param in params]
            gr.on(
                triggers=changes_expressions,
                fn=self.inferencer.edit_expression,
                inputs=params + opt_in_features_params,
                outputs=img_out
            )

            btn_gen.click(self.inferencer.edit_expression,
                          inputs=params + opt_in_features_params,
                          outputs=img_out)

        self.app.launch(inbrowser=True)


app = App()
app.launch()




