import gradio as gr
from gradio_i18n import Translate, gettext as _

from modules.live_portrait.live_portrait_inferencer import LivePortraitInferencer
from modules.utils.paths import *


class App:
    def __init__(self,
                 args=None):
        self.args = args
        self.app = gr.Blocks()
        self.i18n = Translate(I18N_YAML_PATH)
        self.inferencer = LivePortraitInferencer()

    @staticmethod
    def create_parameters():
        return [
            gr.Slider(label=_("Rotate Pitch"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Rotate Yaw"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Rotate Roll"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Blink"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Eyebrow"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Wink"), minimum=0, maximum=25, step=0.5, value=0),
            gr.Slider(label=_("Pupil X"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("Pupil Y"), minimum=-20, maximum=20, step=0.5, value=0),
            gr.Slider(label=_("AAA"), minimum=-30, maximum=120, step=1, value=0),
            gr.Slider(label=_("EEE"), minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label=_("WOO"), minimum=-20, maximum=20, step=0.2, value=0),
            gr.Slider(label=_("Smile"), minimum=-2.0, maximum=2.0, step=0.01, value=0),
            gr.Slider(label=_("Source Ratio"), minimum=0, maximum=1, step=0.01, value=1),
            gr.Slider(label=_("Sample Ratio"), minimum=-0.2, maximum=1.2, step=0.01, value=1),
            gr.Dropdown(label=_("Sample Parts"),
                        choices=["OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"], value="All"),
            gr.Slider(label=_("Crop Factor"), minimum=1.5, maximum=2.5, step=0.1, value=1.7)
        ]

    def launch(self):
        with self.app:
            with self.i18n:
                with gr.Row():
                    with gr.Column():
                        img_ref = gr.Image(label=_("Reference Image"))
                with gr.Row():
                    btn_gen = gr.Button("GENERATE", visible=False)
                with gr.Row():
                    with gr.Column(scale=8):
                        img_out = gr.Image(label=_("Output Image"))
                    with gr.Column(scale=2):
                        expression_parameters = self.create_parameters()
                        btn_openfolder = gr.Button('📂')
                        with gr.Accordion("Opt in features", visible=False):
                            img_sample = gr.Image()
                            img_motion_link = gr.Image()
                            tb_exp = gr.Textbox()

                params = expression_parameters + [img_ref]
                opt_in_features_params = [img_sample, img_motion_link, tb_exp]

                gr.on(
                    triggers=[param.change for param in params],
                    fn=self.inferencer.edit_expression,
                    inputs=params + opt_in_features_params,
                    outputs=img_out
                )

                btn_openfolder.click(
                    fn=lambda: self.open_folder(self.args.output_dir), inputs=None, outputs=None
                )
                btn_gen.click(self.inferencer.edit_expression,
                              inputs=params + opt_in_features_params,
                              outputs=img_out)

            self.app.launch(inbrowser=True)

    @staticmethod
    def open_folder(folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"The directory path {folder_path} has newly created.")
        os.system(f"start {folder_path}")


app = App()
app.launch()




