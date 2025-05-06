import gradio as gr
import os
import tempfile
import base64
from model import predict_with_interpretable_probabilities
from flask import send_file
from io import BytesIO

def analyze_voice(audio):
    """Process the audio and return results"""
    # Save the audio to a temporary file
    temp_dir = './temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', dir=temp_dir, delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Save the audio file received from Gradio
        audio[1].save(temp_path)
        
        # Process audio using your existing model
        predicted_class_label, risk_level, pdf_buffer = predict_with_interpretable_probabilities(temp_path)
        
        # Create report path
        report_dir = './reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f'voice_analysis_report_{os.path.basename(temp_path)}.pdf')
        
        # Save the PDF report
        with open(report_path, 'wb') as f:
            f.write(pdf_buffer.getvalue())
        
        # Convert the PDF to base64 for download
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8')
        pdf_download = f'<a href="data:application/pdf;base64,{pdf_base64}" download="report.pdf">Download Report</a>'
        
        # Create the result message
        if predicted_class_label != "Healthy":
            result = f"Your voice shows {risk_level} indication of {predicted_class_label}"
        else:
            result = "Your voice appears to be healthy"
            
        return result, pdf_download
    
    except Exception as e:
        return f"Error processing audio: {str(e)}", None
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Create the Gradio interface
with gr.Blocks(title="Voice Analysis Model") as demo:
    gr.Markdown("# Voice Analysis Model")
    gr.Markdown("Upload an audio recording of your voice to analyze potential health conditions.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
            analyze_btn = gr.Button("Analyze Voice")
        
        with gr.Column():
            result_output = gr.Textbox(label="Analysis Result")
            report_output = gr.HTML(label="Report")
    
    analyze_btn.click(
        fn=analyze_voice,
        inputs=[audio_input],
        outputs=[result_output, report_output]
    )
    
    gr.Markdown("## How to use")
    gr.Markdown("""
    1. Record your voice using the microphone or upload an audio file
    2. Click 'Analyze Voice'
    3. View the analysis result and download the detailed report
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 