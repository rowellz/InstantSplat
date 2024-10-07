from instant_splat.gradio_ui.multi_img_ui import multi_img_block
import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab(label="Multi Image"):
        multi_img_block.render()

if __name__ == "__main__":
    demo.launch()
