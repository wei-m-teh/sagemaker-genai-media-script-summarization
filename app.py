import gradio as gr
import ai21
import boto3
import os
import time

textract = boto3.client("textract")
s3 = boto3.client("s3")
s3_bucket = "sagemaker-us-east-1-602900100639"
s3_input_prefix = "data/script-summarization/inputs"
s3_output_prefix = "data/script-summarization/outputs"

def extract_text(file_path):
    basename = os.path.basename(file_path)
    response = textract.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket,
                'Name': os.path.join(s3_input_prefix, basename)
            }
        },
        OutputConfig={
            'S3Bucket': s3_bucket,
            'S3Prefix': s3_output_prefix
        },
    )
    job_id = response['JobId']
    get_job_response = textract.get_document_text_detection(JobId=job_id)
    status = get_job_response['JobStatus']

    while status != "SUCCEEDED":
        get_job_response = textract.get_document_text_detection(JobId=job_id)
        status = get_job_response['JobStatus']

    all_blocks = []
    blocks = get_job_response['Blocks']
    all_blocks.extend(blocks)
    while 'NextToken' in get_job_response and len(get_job_response['NextToken']) > 0:
        print(f"processing next_token: {get_job_response['NextToken']}")
        get_job_response = textract.get_document_text_detection(JobId=job_id, NextToken=get_job_response['NextToken'])
        blocks = get_job_response['Blocks']
        all_blocks.extend(blocks)

    return all_blocks

def upload_file(file_path):
    with open(file_path, "rb") as file_obj:
        basename = os.path.basename(file_path)
        s3.upload_fileobj(file_obj, s3_bucket, f"{s3_input_prefix}/{basename}")
        return f"s3://{s3_bucket}/{s3_input_prefix}/{basename}"


def format_text(texts):
    header = None
    formatted_lines = []
    dialog_lines = []
    for idx, line in enumerate(texts):
        if line.isupper():
            if len(dialog_lines) > 0:
                dialog_lines.insert(0, f"{header.strip()}:")
                formatted_lines.append(dialog_lines)
                dialog_lines = []
            header = line
        else:
            open_parenthesis_idx = line.find("(")
            close_parenthesis_idx = line.find(")")
            if open_parenthesis_idx != -1 and close_parenthesis_idx != -1:
                temp_line = line[:open_parenthesis_idx] + line[close_parenthesis_idx + 1:]
                if temp_line.strip().isupper():
                    if len(dialog_lines) > 0:
                        dialog_lines.insert(0, f"{header.strip()}:")
                        formatted_lines.append(dialog_lines)
                        dialog_lines = []
                    header = line
            else:
                if header:  # This will skip lines that are not part of the plot. e.g. title, or author etc.
                    dialog_lines.append(line.strip())

    dialog_lines.insert(0, f"{header.strip()}:")
    dialog_lines.append(line.strip())  # final line
    formatted_lines.append(dialog_lines)
    return formatted_lines


def process_file(file_obj):
    s3_file_path = upload_file(file_obj.name)
    blocks = extract_text(s3_file_path)
    texts = []
    for block in blocks:
        if block['BlockType'] == 'LINE':
            text = block['Text']
            texts.append(text)
    formatted_texts = format_text(texts)
    formatted_lines = ""
    for formatted_text in formatted_texts:
        formatted_lines += " ".join(formatted_text) + "\n"
    return formatted_lines


def generate_summary(prompt, numResults=1, temperature=0.1, max_tokens=70):
    response = ai21.Completion.execute(sm_endpoint="j2-jumbo-instruct",
                                       prompt=prompt,
                                       maxTokens=max_tokens,
                                       temperature=temperature,
                                       numResults=numResults)
    summaries = []
    for completion in response['completions']:
        summaries.append(completion['data']['text'][1:])

    return summaries

def summarize_script(script, temperature, prompt, lines_per_scene, strides, max_tokens):
    lines = script.split("\n")
    summaries = []
    for starting_line in range(0, len(lines), int(strides)):
        lines_to_be_summarized = lines[starting_line: starting_line + int(lines_per_scene)]
        prompt_ = " ".join(lines_to_be_summarized) + "\n\n" + "As a screenwriter, write me one sentence that summarizes the movie scene above"
        summary = generate_summary(prompt_, 1, 0)
        summaries.extend(summary)

    lines = summaries
    summarized_prompt = " ".join(lines) + "\n\n" + prompt
    summary = generate_summary(summarized_prompt, 1, temperature)
    return summary[0]

with gr.Blocks() as demo:
    gr.Markdown("Upload a movie/show script in PDF or TXT file")
    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.File(file_types=["pdf", "txt"])
            btn = gr.Button("Upload")
        with gr.Column(scale=2):
            out = gr.Textbox(label="Extracted Script", interactive=True)
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt:",
                                max_lines=1)
            temperature = gr.Slider(0.0, 1.0, value=0, step=0.1, label="Temperature", info="Choose between 0.0 and 1.0")
            lines_per_scene = gr.Number(label="Lines to summarize per run", value=100)
            strides = gr.Number(label="Strides length", value=70)
            max_tokens = gr.Number(label="Maximum number of tokens (words) in the output", value=70)
            btn_summary = gr.Button("Summarize It")
            script_summary_output = gr.Textbox(label="Script Summary")

    btn.click(fn=process_file, inputs=file_uploader, outputs=out)
    btn_summary.click(fn=summarize_script,
                      inputs=[out, temperature, prompt, lines_per_scene, strides, max_tokens],
                      outputs=script_summary_output)

if 'GRADIO_USERNAME' in os.environ and 'GRADIO_PASSWORD' in os.environ:
    demo.launch(server_name="0.0.0.0", share=False, auth=(os.environ['GRADIO_USERNAME'], os.environ['GRADIO_PASSWORD']))
else:
    demo.launch(server_name="0.0.0.0", share=False)
