from langchain_community.llms import Ollama
import transformers
import torch
llm = Ollama(model="llama3:8b-instruct")
#model_id = ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"]
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

import os
os.environ["HF_TOKEN"]="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"
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

import time
from time import time

result_dir = "Result"

if not(os.path.isdir(result_dir)):
    os.mkdir(result_dir)

from discord_sender import send_message
from datetime import datetime
from time import sleep

def alive(_worker_id):
    second = 1
    minute = 60 * second
    hour = 60 * minute
    while True:
        sleep(30 * second)
        send_message("worker_{} is alive".format(_worker_id))

from multiprocessing import Process

def model(_model_id, _worker_id, datasets):
    proc = Process(target=alive, args=(_worker_id, ))
    proc.start()
    _model = _model_id
    print("START: {} from worker_{}\nPID: {}".format(_model, _worker_id, os.getpid()))
    nowtime = datetime.now().strftime("%Y %m %d %H:%M:%S")
    send_message("START: {} from worker_{}".format(_model, _worker_id))

    start = time()
    pipeline = transformers.pipeline(
        "text-generation",
        model=_model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    message = None
    if _worker_id % 2 == 0:
        messages = [
            {"role": "system", "content": "You are a Opensource expert who can "},
            {"role": "user", "content": datasets['train']},
            {"role": "user", "content": "Where is this code from"+datasets['train'][0]['instruction']}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a Opensource expert who can "},
            {"role": "user", "content": "Where is this code from"+datasets['train'][0]['instruction']}
        ]
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    end = time()
    proc.kill()
    proc.join()
    print("FINISH: {} from worker{}".format(_model, _worker_id))
    send_message("FINISH: {} from worker_{}.\nExecution Time: {}".format(_model, _worker_id, end - start))
    yield(outputs[0]["generated_text"][len(prompt):], end - start, _model, nowtime)

result_file = "Result"
ext = ".txt"

def trainer(worker_id, datasets):
    for result, execution_time, used_model, start_time in model(model_id, worker_id, datasets):
        
        res_dict = dict()
        res_dict['Result'] = result
        res_dict['Time'] = execution_time
        result_path = os.path.join(result_dir, result_file+str(worker_id)+ext)

        with open(result_path, 'a') as f:
            f.write("Start at: {}\nModel: {}\nTime: {}\nResult: {}\n\n\n".format(start_time, used_model, res_dict['Time'], res_dict['Result']))