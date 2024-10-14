from instant_splat.gradio_ui.multi_img_ui import multi_img_block
import gradio as gr

title = "# Instant Splat"
description1 = """InstantSplat Demo using Rerun and Gradio"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    with gr.Tab(label="Multi Image"):
        multi_img_block.render()

if __name__ == "__main__":
    demo.launch()
