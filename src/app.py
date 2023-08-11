# %%
# Inspired in:
# https://www.analyticsvidhya.com/blog/2023/05/build-a-chatgpt-for-pdfs-with-langchain/
# https://github.com/sunilkumardash9/Pdf-GPT/blob/main/app.py


import gradio as gr
import fitz
from PIL import Image
from utils import DocumentChatApp
                  

# ============= Initialize LLM App =============

app = DocumentChatApp()


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



# ============= Build Chat Interface =============

with gr.Blocks() as demo:

    # ============= Create a Gradio block =============
    with gr.Column():
        with gr.Row():
            chatbot = gr.Chatbot(value=[],
                                 elem_id='chatbot'
                                 ).style(height=650)
            show_img = gr.Image(label='Upload File',
                                tool='select'
                                ).style(height=680)

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press enter"
            ).style(container=False)

        with gr.Column(scale=0.15):
            submit_btn = gr.Button('Submit')

        with gr.Column(scale=0.15):
            btn = gr.UploadButton("📁 Upload a file",
                                   file_types=[app.loader_map.keys()]
                                  ).style()

    # ============= Set up event handlers =============

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


demo.queue()


# %%

if __name__ == "__main__":
    demo.launch(share=True)

