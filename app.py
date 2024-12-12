import gradio as gr
from datetime import datetime
import unicodedata
from utils.utils import get_label



def sanitize_feedback(feedback):
    """
    Convert emojis or other non-text characters in feedback to a text representation.
    This ensures only clean text is saved in the feedback file.
    """
    sanitized_feedback = ''.join(
        c if unicodedata.category(c).startswith(('L', 'N', 'P', 'Z')) else '' for c in feedback
    )
    return sanitized_feedback


# Function to capture feedback and save it to a timestamped file
def capture_feedback(feedback):
    # Sanitize feedback to ensure only text is saved
    sanitized_feedback = sanitize_feedback(feedback)
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a filename with the timestamp
    filename = f"{timestamp}.txt"
    # Write feedback to the file
    with open(filename, "w", encoding="utf-8") as file:
        file.write(sanitized_feedback)
    return "Thank you for your feedback!"   # Return message for the popup


# Main Gradio interface
with gr.Blocks() as demo:
    # State to manage interface visibility
    feedback_submitted = gr.State(False)

    # Title and description of the demo
    gr.Markdown("<h1 style='text-align: center;'>Emotions Classification Demo</h1>")
    gr.Markdown("""
    <div style='text-align: center;'>
        This is a demo page for our classification model. Our model provides a brief description of your image and predicts the most relevant emotions that the image invokes.
    </div>
    """)

    # Organize the input and output sections in a row
    with gr.Row():
        # Left side: input section with description
        with gr.Column():
            gr.Markdown("<b>Upload your image here to get the Emotion predictions.</b>")
            image_input = gr.Image(type="pil", label="Input Image")

            # Submit button for image upload
            submit_button = gr.Button("Submit")

        # Right side: output section with description
        with gr.Column():
            gr.Markdown("<b>Predicted outputs</b>")
            # Textbox for emotion label
            output_text = gr.Textbox(label="Image Description")
            # Plot for the bar chart
            output_plot = gr.Plot(label="Emotion Probabilities")

            # Feedback section directly below the graph
            gr.Markdown("<b>How do you feel about our emotion results?</b>")
            feedback_choices = [
                "😄 Very Satisfied",
                "😊 Satisfied",
                "😐 Neutral",
                "🙁 Dissatisfied",
                "😡 Very Dissatisfied"
            ]
            feedback = gr.Radio(choices=feedback_choices, label="Your Feedback")

            # Button to submit feedback
            feedback_button = gr.Button("Submit Feedback")


    # Feedback submission action
    def submit_feedback(feedback):
        capture_feedback(feedback)  # Save feedback to file
        feedback_submitted.set(True)  # Set state to indicate feedback was submitted
        return "Thank you for your feedback!"  # Return thank-you message


    feedback_button.click(fn=submit_feedback, inputs=feedback, outputs=None)

    # Thank you message section, initially hidden
    thank_you_message = gr.Markdown("", visible=False)

    # Display thank you message when feedback is submitted
    feedback_submitted.change(
        fn=lambda: ("Thank you for your feedback!", False),  # Show thank you message
        inputs=feedback_submitted,
        outputs=[thank_you_message, feedback_submitted]
    )

    # Main interface to show
    with gr.Row(visible=True):
        gr.Markdown("Thank you for your feedback!", visible=feedback_submitted)

    # Function to process the input and output
    submit_button.click(fn=get_label, inputs=image_input, outputs=[output_text, output_plot])

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
