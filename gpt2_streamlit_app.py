import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Streamlit app
st.title("Text Generation with GPT2")

# Input text
input_text = st.text_area("Enter your prompt:", "Once upon a time")

# Parameters in the sidebar
st.sidebar.title("Tune the Parameters")
max_length = st.sidebar.slider(
    "Max Length", 
    min_value=10, 
    max_value=200, 
    value=50, 
    help="The maximum length of the generated text, including the input text."
)
temperature = st.sidebar.slider(
    "Temperature", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.7, 
    help="Controls the randomness of predictions by scaling the logits before applying softmax. Lower values make the model more confident and deterministic, while higher values increase randomness."
)
top_k = st.sidebar.slider(
    "Top K", 
    min_value=0, 
    max_value=100, 
    value=50, 
    help="Limits the sampling pool to the top K tokens with the highest probabilities. This helps in reducing the likelihood of generating less probable tokens."
)
top_p = st.sidebar.slider(
    "Top P", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.9, 
    help="Limits the sampling pool to the smallest set of tokens whose cumulative probability is greater than or equal to P. This helps in balancing between deterministic and random sampling."
)
repetition_penalty = st.sidebar.slider(
    "Repetition Penalty", 
    min_value=1.0, 
    max_value=2.0, 
    value=1.2, 
    help="Penalizes repeated tokens to reduce the likelihood of repetitive text generation."
)
num_return_sequences = st.sidebar.slider(
    "Number of Sequences", 
    min_value=1, 
    max_value=5, 
    value=1, 
    help="The number of different sequences to generate."
)


# Generate text button
if st.button("Generate Text"):
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate text with fine-tuned parameters
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True
    )

    # Decode and display the generated text
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        st.write(generated_text)

# Contact details
st.markdown("""
    ---
    I'd love your feedback :smiley: Want to collaborate? Develop a project? Find me on [LinkedIn](https://www.linkedin.com/in/lopezyse/), [X](https://x.com/lopezyse) and [Medium](https://lopezyse.medium.com/)
""")

# Run the app with: streamlit run gpt2_streamlit_app.py