import gradio as gr
from chat import RAGChat
from embed import RAGCreate
from typing import Tuple, List, Optional

class RAGUI:
    def __init__(self):
        self.rag_chat = None
        self.rag_embed = None
        self.theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.slate,
            secondary_hue=gr.themes.colors.purple
        )

    def create_vector_store(
        self, file_path_list: list, 
        embedding_model: str, 
        nim_api: str
    ) -> str:
        """Create vector store from uploaded document"""
        try:
            self.rag_embed = RAGCreate(
                filepath=file_path_list,
                model_name=embedding_model,
                nim_api=nim_api
            )
            self.rag_embed.create_collection()
            return "Vector store created successfully!"
        except Exception as e:
            return f"Error creating vector store: {str(e)}"

    def load_llm_model(
        self, 
        llm_model: str,
        embedding_model: str,
        nim_api: str,
        system_prompt: str
    ) -> str:
        """Load the selected LLM model"""
        try:
            self.rag_chat = RAGChat(
                llm_model=llm_model,
                embedding_model=embedding_model,
                nim_api=nim_api,
                system_prompt=system_prompt
            )
            return f"Model {llm_model} loaded successfully!"
        except Exception as e:
            return f"Error loading model: {str(e)}"

    def chat_response(
        self,
        chatbot: List[Tuple[str, str]],
        question: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """Generate chat response"""
        if not self.rag_chat or not self.rag_embed:
            return chatbot + [(question, "Please load a model and create vector store first.")], ""
        
        try:
            response = self.rag_chat.answer_question(
                question=question,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            chatbot.append((question, response))
            return chatbot, ""
        except Exception as e:
            return chatbot + [(question, f"Error: {str(e)}")], ""

    def create_ui(self):
        """Create the Gradio UI interface"""
        with gr.Blocks(theme=self.theme) as demo:
            gr.Markdown("# Retrieval Augmented Generation")
            
            nim_api = gr.Textbox(label='Enter your NIM API key', type="password")

            with gr.Row():
                # Left Column - Document Upload and Embedding
                with gr.Column(scale=0.5, variant='panel'):
                    gr.Markdown("## Upload Document & Select the Embedding Model")
                    file = gr.File(type="filepath", file_count="multiple")
                    
                    with gr.Row():
                        embedding_model = gr.Dropdown(
                            choices=[
                                "nvidia/nv-embedqa-e5-v5",
                                "nvidia/llama-3.2-nv-embedqa-1b-v2",
                                "nvidia/embed-qa-4"
                            ],
                            label="Select the embedding model"
                        )
                        vector_index_btn = gr.Button('Create vector store', variant='primary')
                        vector_index_msg = gr.Textbox(show_label=False, lines=1)

                    system_prompt = gr.Textbox(
                        label="System instruction",
                        lines=3,
                        value="Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only. If you do not find any information related to the question in the given context, just say that you don't know."
                    )
                    
                    with gr.Accordion("Text generation parameters"):
                        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                        max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=2048, value=512, step=1)
                        top_p = gr.Slider(label="Top P", minimum=0, maximum=1, value=0.95, step=0.05)

                # Right Column - Chat Interface
                with gr.Column(scale=0.5, variant='panel'):
                    gr.Markdown("## Chatbot")
                    
                    with gr.Row():
                        llm_model = gr.Dropdown(
                            choices=["local",
                                      "meta/llama-3.3-70b-instruct",
                                      "qwen/qwen2.5-7b-instruct",
                                      "google/gemma-2-27b-it",
                                      "mistralai/mistral-small-24b-instruct"],
                            value="local",
                            label="Select the LLM"
                        )
                        model_load_btn = gr.Button('Load Model', variant='primary')
                        model_status = gr.Textbox(show_label=False, lines=1)

                    chatbot = gr.Chatbot([], elem_id="chatbot", height=725)
                    question = gr.Textbox(
                        label="Question",
                        lines=2,
                        placeholder="Enter your question and press Enter"
                    )

                    with gr.Row():
                        submit_btn = gr.Button('Submit', variant='primary')
                        clear_btn = gr.Button('Clear', variant='stop')

            # Event handlers
            vector_index_btn.click(
                fn=self.create_vector_store,
                inputs=[file, embedding_model, nim_api],
                outputs=vector_index_msg
            )

            model_load_btn.click(
                fn=self.load_llm_model,
                inputs=[llm_model, embedding_model, nim_api, system_prompt],
                outputs=model_status
            )

            submit_btn.click(
                fn=self.chat_response,
                inputs=[
                    chatbot, question, temperature, max_new_tokens, top_p
                ],
                outputs=[chatbot, question]
            )

            clear_btn.click(lambda: None, None, chatbot, queue=False)

        return demo

def main():
    ui = RAGUI()
    demo = ui.create_ui()
    demo.launch(debug=True, share=True)

if __name__ == "__main__":
    main()