import gradio as gr
import boto3
import os
import time
import logging
import json
import botocore

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)

textract = boto3.client("textract")
s3 = boto3.client("s3")
s3_bucket = os.environ.get("S3_BUCKET", "sagemaker-us-east-1-602900100639")
s3_input_prefix = os.environ.get("S3_BUCKET_PREFIX", "data/script-summarization/inputs")
s3_output_prefix = os.environ.get("S3_BUCKET_OUTPUT_PREFIX", "data/script-summarization/outputs")
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-v2")
BEDROCK_SERVICE_MAX_RETRIES = 10
def extract_text(file_path):
    logging.info("Extract Text")
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
    logging.info(f"JobId: {job_id}")
    get_job_response = textract.get_document_text_detection(JobId=job_id)
    status = get_job_response['JobStatus']

    latest_get_job_response = None
    while status != "SUCCEEDED":
        get_job_response = textract.get_document_text_detection(JobId=job_id)
        logging.info(f"job status: {get_job_response}")
        status = get_job_response['JobStatus']
        latest_get_job_response = get_job_response

    logging.info(f"job {job_id} completed successfully")
    all_blocks = []
    blocks = latest_get_job_response['Blocks']
    all_blocks.extend(blocks)
    while 'NextToken' in get_job_response and len(get_job_response['NextToken']) > 0:
        logging.info(f"processing next_token: {get_job_response['NextToken']}")
        get_job_response = textract.get_document_text_detection(JobId=job_id, NextToken=get_job_response['NextToken'])
        blocks = get_job_response['Blocks']
        all_blocks.extend(blocks)

    return all_blocks

def upload_file(file_path):
    logging.info("Uploading file")
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

example_config_dict = {
   "metal-heart.txt" : { "temperature" : 0, "chunk_size": 50, "overlap" : 10, "max_tokens" : 50},
   "wings-of-light.txt" : { "temperature" : 0, "chunk_size": 50, "overlap" : 10, "max_tokens" : 50},
   "the-journey-within.txt" : { "temperature" : 0, "chunk_size": 50, "overlap" : 10, "max_tokens" : 50}
}
def process_file(file_obj):
    logging.info("Processing file")
    temperature_comp = 0
    chunk_size_comp = 100
    overlap_comp = 30
    max_tokens_comp = 50
    is_example = False
    if isinstance(file_obj, str):  # coming from the example
        filename = f"examples/{file_obj}"
        is_example=True
        if file_obj.strip() in example_config_dict:
            temperature_comp = example_config_dict[file_obj.strip()]['temperature']
            chunk_size_comp = example_config_dict[file_obj.strip()]['chunk_size']
            overlap_comp = example_config_dict[file_obj.strip()]['overlap']
            max_tokens_comp = example_config_dict[file_obj.strip()]['max_tokens']
    else:
        filename = file_obj.name
    extension = filename.split(".")[-1]
    if extension.lower() == "txt" or extension.lower() == "TXT":
        with open(filename, "r") as f:
            data = f.readlines()
            formatted_lines = "".join(data)
            if is_example:
                return formatted_lines, temperature_comp, chunk_size_comp, overlap_comp, max_tokens_comp
            else:
                return formatted_lines
    if extension.lower() == "pdf" or extension.lower() == "PDF":  # Examples are always in txt, not in PDF
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
    else:
        raise gr.Error("Unsupported file type. Only pdf or txt files are supported")

def get_bedrock_client():
    if "ASSUMABLE_ROLE_ARN" in os.environ:
        session = boto3.Session()
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=os.environ.get("ASSUMABLE_ROLE_ARN", None),
            RoleSessionName="bedrock"
        )
        new_session = boto3.Session(aws_access_key_id=response['Credentials']['AccessKeyId'],
                              aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                              aws_session_token=response['Credentials']['SessionToken'])

        bedrock = new_session.client('bedrock' , 'us-east-1')
    else:
        bedrock = boto3.client("bedrock", "us-east-1")
    return bedrock

boto3_bedrock = get_bedrock_client()
def generate_summary(prompt, temperature=0.1, max_tokens=70):
    global boto3_bedrock

    body = json.dumps({
        "prompt": prompt, "max_tokens_to_sample": max_tokens, "temperature": temperature,
        "top_k": 250, "top_p": 1.0, "stop_sequences": []}
    )
    content_type = "application/json"
    try:
        response = boto3_bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept="*/*", contentType=content_type)
    except botocore.exceptions.ClientError as err:
        if err.response['Error']['Code'] == "ExpiredToken":
            print("Expired token. Generate a new token and retry")
            boto3_bedrock = get_bedrock_client()
            response = boto3_bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept="*/*",
                                                  contentType=content_type)
        elif err.response['Error']['Code'] == 'ThrottlingException':
            print("Bedrock service is being throttled")
            call_nbr = 0
            while call_nbr < BEDROCK_SERVICE_MAX_RETRIES:
                print("Bedrock service is being throttled, retrying..")
                time.sleep(1)
                try:
                    response = boto3_bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept="*/*",
                                               contentType=content_type)
                    response_body = json.loads(response.get('body').read())
                    summaries = []
                    summaries.append(response_body['completion'])
                    return summaries
                except:
                    call_nbr += 1
            print("Bedrock service is being throttled, maximum retries reached..throws exception")
            raise gr.Error("Service is at capacity right now, please try again later")

    response_body = json.loads(response.get('body').read())
    summaries = []
    summaries.append(response_body['completion'])
    return summaries

def summarize_script(script, temperature, prompt, chunk_size, overlap, max_tokens):
    lines = script.split("\n")
    strides = chunk_size - overlap
    summaries = []
    for starting_line in range(0, len(lines), int(strides)):
        lines_to_be_summarized = lines[starting_line: starting_line + int(chunk_size)]
        prompt_ = "\n\nHuman: Given the movie scene below wrapped in <scene></scene> tags:" + "\n" + "\n<scene>\n" + " ".join(lines_to_be_summarized) + "\n</scene>\n" + \
                  "As a screenwriter, summarizes the movie scene in one sentence." + "\n\nAssistant:"
        summary = generate_summary(prompt_, 0)
        summaries.extend(summary)

    lines = summaries
    summarized_prompt = "\n\nHuman: Given the movie scene below wrapped in <scene></scene> tags:\n" + \
                        "\n<scene>\n" + " ".join(lines) + "\n</scene>\n" + prompt + "\nOnly show the summary text.\n\nAssistant:"
    summary = generate_summary(summarized_prompt, temperature)
    return summary[0]

default_temperature = gr.Slider(0.0, 1.0, value=0, step=0.1, label="Temperature", info="Choose between 0.0 and 1.0 to control the randomness in the output. "
                                                                                       "A high temperature produces more creative results; "
                                                                                       "A low temperature produces more conservative results.")
default_chunk_size = gr.Number(label="Lines to summarize per chunk", value=100, info='Number of lines to include in generating a summary')
default_overlap = gr.Number(label="Lines to overlap from previous chunk", value=30, info="Number of lines from previous chunk to include in generating a summary")
default_max_tokens = gr.Number(label="Maximum Generated tokens (words)", value=70)
default_out = gr.Textbox(label="Extracted Screenplay (Editable)", interactive=True)

with gr.Blocks(theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg,
                                       font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('![](file/img/AWS-MnE.jpeg)')
        with gr.Column(scale=2):
            gr.Markdown('<h1 class="text10xl font-bold font-secondary text-gray-800 dark:text-gray-100 uppercase">'
                    '<p style="text-align: left; vertical-align: bottom;"><font size="+3">MEDIA SCREENPLAY AI ASSISTANT</font></p></h1>')
            gr.Markdown('<p style="text-align: left; vertical-align: bottom;padding: 2px 50px;font-family: verdana; color:grey"><font size="+1"><br>Create summary, synopsis or EPG in seconds</br></font></p>')

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                file_uploader = gr.File(file_types=["pdf", "txt"], label="Upload a pdf or text, or try some examples",
                                        show_label=True)
                gr.Examples(
                    label="Example Screenplays",
                    examples=["metal-heart.txt", "wings-of-light.txt", "the-journey-within.txt"],
                    inputs=[default_out],
                    outputs=[default_out, default_temperature, default_chunk_size, default_overlap, default_max_tokens],
                    fn=process_file,
                    cache_examples=False,
                    run_on_click=True
                )
            out = default_out.render()
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt:",
                                max_lines=1,
                                value="As a screenwriter, write me one sentence that summarizes the given movie script.",
                                interactive=True)
            temperature = default_temperature.render()
            chunk_size = default_chunk_size.render()
            overlap = default_overlap.render()
            max_tokens = default_max_tokens.render()
            btn_summary = gr.Button("Summarize it!", variant="primary")
            script_summary_output = gr.Textbox(label="Screenplay Summary")
            gr.Markdown(
        """
        # Chunking Strategy Illustration
        ![](file/img/stride-explain-2.png)
        """)
    '''
    with gr.Row():
        gr.Markdown(
    """
    # Chunking strategy
    ![](file/img/stride-explain.png)
    """)
    '''
    btn_summary.click(fn=summarize_script,
                      inputs=[out, temperature, prompt, chunk_size, overlap, max_tokens],
                      outputs=script_summary_output)
    file_uploader.upload(fn=process_file, inputs=file_uploader, outputs=[out])

if 'GRADIO_USERNAME' in os.environ and 'GRADIO_PASSWORD' in os.environ:
    demo.queue().launch(server_name="0.0.0.0", share=False,
                        auth=(os.environ['GRADIO_USERNAME'],
                              os.environ['GRADIO_PASSWORD']))
else:
    demo.queue().launch(server_name="0.0.0.0", share=False)
