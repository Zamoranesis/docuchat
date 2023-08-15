# %%
# Inspired in:
# https://www.analyticsvidhya.com/blog/2023/05/build-a-chatgpt-for-pdfs-with-langchain/
# https://github.com/sunilkumardash9/Pdf-GPT/blob/main/app.py


import gradio as gr
import fitz
from PIL import Image
from utils import DocumentChatApp
import json
                  

# ============= Initialize LLM App =============

app = DocumentChatApp()
app.model_config

# %%

# ============= Auxiliary functions =============


def add_text(history, text: str):
        if not text:
            raise gr.Error('enter text')
        history = history + [(text,'')] 

        return history


def get_response(history,
                 query,
                 file
                 ): 
        if not file:
            raise gr.Error(message='Upload a file')  

        chain = app(file)
        result = chain({"question": query,
                        'chat_history':app.chat_history
                        },
                        return_only_outputs=True
                        )
        app.chat_history += [(query, result["answer"])]
        app.N = list(result['source_documents'][0])[1][1]['page']
        for char in result['answer']:
           history[-1][-1] += char

           yield history,''


def render_file(file):
        doc = fitz.open(file.name)
        page = doc[app.N]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

        return image


def render_first(file):
        doc = fitz.open(file)
        page = doc[0]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB',
                                 [pix.width, pix.height],
                                  pix.samples
                                  )

        return image


def overwrite_model_config( n_ctx_input,
                            n_batch_input,
                            n_gpu_layers_input,
                            max_tokens_input,
                            temperature_input,
                            top_p_input,
                            repeat_penalty_input
                            ):

        print(f"Previous model config:\n{app.model_config}")
        # Overwrite model config
        app.model_config["n_ctx"] = int(n_ctx_input)
        app.model_config["n_batch"] = int(n_batch_input)
        app.model_config["n_gpu_layers"] = int(n_gpu_layers_input)
        app.model_config["max_tokens"] = int(max_tokens_input)
        app.model_config["temperature"] = float(temperature_input)
        app.model_config["top_p"] = float(top_p_input)
        app.model_config["repeat_penalty"] = float(repeat_penalty_input)
        print(f"New model config:\n{app.model_config}")


def overwrite_vectordb_config(chunk_size_input,
                              chunk_overlap_input,
                              k_input
                             ):
        print(f"Previous vectordb config:\n{app.vectordb_config}")
        # Overwrite model config
        app.vectordb_config["chunk_size"] = int(chunk_size_input)
        app.vectordb_config["chunk_overlap"] = int(chunk_overlap_input)
        app.vectordb_config["k"] = int(k_input)
        print(f"New vectordb config:\n{app.vectordb_config}")


# ============= Build Chat Interface =============

with gr.Blocks() as demo:

    with gr.Tabs():
        
        # Pestaña de Chat
        with gr.Tab("Chat"):
            # Bloque de chatbot y imagen
            with gr.Column():
                with gr.Row():
                    chatbot = gr.Chatbot(value=[],
                                        elem_id='chatbot',
                                        height=650
                                        )
                    show_img = gr.Image(label='Upload File',
                                        tool='select',
                                        height=680
                                        )

            with gr.Row():
                with gr.Column(scale=0.70):
                    txt = gr.Textbox(
                                    show_label=False,
                                    placeholder="Enter text and press enter",
                                    container=False
                                    )

                with gr.Column(scale=0.15):
                    submit_btn = gr.Button('Submit')

                with gr.Column(scale=0.15):
                    btn = gr.UploadButton("📁 Upload a file",
                                        file_types=[app.loader_map.keys()]
                                        )
            # Event handler for uploading a PDF
            btn.upload(fn=render_first,
                    inputs=[btn],
                    outputs=[show_img]
                    )
            # Event handler for submitting text and generating response
            submit_btn.click(
                fn=add_text,
                inputs=[chatbot, txt],
                outputs=[chatbot],
                queue=False
            ).success(
                fn=get_response,
                inputs=[chatbot, txt, btn],
                outputs=[chatbot, txt]
            ).success(
                fn=render_file,
                inputs=[btn],
                outputs=[show_img]
            )


        # Pestaña de Configuración
        with gr.Tab("Configuration"):
            with gr.Column():
                with gr.Row():
                    n_ctx_input = gr.Number(value=1028, label="N CTX")
                    n_batch_input = gr.Number(value=512, label="N Batch")
                    n_gpu_layers_input = gr.Number(value=128, label="N GPU Layers")
                    max_tokens_input = gr.Number(value=512, label="Max Tokens")
                    temperature_input = gr.Slider(value=0.1, minimum=0 , maximum= 1, label="Temperature")
                    top_p_input = gr.Slider(value=0.75, minimum=0 , maximum= 1, label="Top P")
                    repeat_penalty_input = gr.Slider(value=1.1, minimum=0 , maximum= 2, label="Repeat Penalty")
                    save_model_config_btn = gr.Button("Save Model Config")
                    save_model_config_btn.click(fn=overwrite_model_config,
                                                inputs=[
                                                        n_ctx_input,
                                                        n_batch_input,
                                                        n_gpu_layers_input,
                                                        max_tokens_input,
                                                        temperature_input,
                                                        top_p_input,
                                                        repeat_penalty_input
                                                        ]
                                                )

            # Configuración de vectordb_config.json
            with gr.Column():
                with gr.Row():
                    chunk_size_input = gr.Number(value=750, label="Chunk Size")
                    chunk_overlap_input = gr.Number(value=50, label="Chunk Overlap")
                    k_input = gr.Number(value=2, label="K")
                    save_vector_config_btn = gr.Button("Save VectorDB Config")
                    save_vector_config_btn.click(fn=overwrite_vectordb_config,
                                                 inputs=[
                                                         chunk_size_input,
                                                         chunk_overlap_input,
                                                         k_input
                                                        ]
                                                )


demo.queue()


# %%

if __name__ == "__main__":
    demo.launch(share=True)

