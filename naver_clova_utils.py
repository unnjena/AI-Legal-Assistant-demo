
import base64
import json
import time
import http.client
from fuzzysearch import find_near_matches
from tqdm.auto import tqdm


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/completions/LK-C',
                     json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['text']
        else:
            return 'Error'


completion_executor = CompletionExecutor(
    host='clovastudio.apigw.ntruss.com',
    api_key='NTA0MjU2MWZlZTcxNDJiY0sAqwXbXAupQ/JYs/4LzhF2BoUDtVTuB5jJYADpi4WbDN7xdwQvPNtHAbNYG6qQUyheKAqXhWvbETl55JvAIXvcMlBLbH5YcAQFiPJaJqDANHAbKHEACGPrhAdJmU17w1ivIzdcIu9xyw4S0y0xsZLisRjM+SgFT9Qqaab8wytPpvoYhwfO+UNcH0VXUMg/Ig7M8XYncN0K4mk0ED/C6Zw=',
    api_key_primary_val='pgI20kra3YJpQM1yoSt5Oq8C3sqzlgiAPFZL7AAa',
    request_id='c3c55f7264b341c59931d547af9c771a'
)


def request_clova_answer(preset, target):
    input = preset + target + '\n' + '*법률가 답변:'
    request_data = {
        'text': input,
        'maxTokens': 220,
        'temperature': 0.4,
        'topK': 5,
        'topP': 0.8,
        'repeatPenalty': 8.0,
        'start': '',
        'restart': '',
        'stopBefore': ["###"],
        'includeTokens': True,
        'includeAiFilters': True,
        'includeProbs': True
    }
    response = completion_executor.execute(request_data)
    return input, response


def make_answer(preset, target):
    res = request_clova_answer(preset, target)
    if "###" in res[1]:
        res_1 = res[1][len(preset):].split("\n###")[0]
    else:
        res_1 = res[1][len(preset):]
    try:
        res_final = '\n'.join(res_1.split("*법률가 답변:")[1].split('\n')[:-1])
    except:
        return ''
    return res_final


def preprocessing(text):
    while ('\n\n' in text):
        text = text.replace('\n\n', '\n')
    while ('\t\t' in text):
        text = text.replace('\t\t', '\t')
    return text


def answer_generation(question):
    file = open("asset/answer_generation_prompt.txt", "r")
    answer_generation_prompt = ''.join(file.readlines())
    file.close()

    target = preprocessing(question)
    answer = ''
    while (len(answer) == 0):
        answer = make_answer(answer_generation_prompt, target)
    return answer


def request_clova_question(preset, target):
    input = preset + target
    request_data = {
        'text': input,
        'maxTokens': 250,
        'temperature': 0.2,
        'topK': 0,
        'topP': 0.8,
        'repeatPenalty': 7.0,
        'start': '',
        'restart': '',
        # 'stopBefore': ['"\n'],
        'includeTokens': True,
        'includeAiFilters': True,
        'includeProbs': True
    }
    response = completion_executor.execute(request_data)
    return input, response


def make_question(crime, preset):
    res = request_clova_question(preset, f"{crime}죄")
    if "###" in res[1]:
        res_1 = res[1][len(preset):].split("\n###")[0]
    else:
        res_1 = res[1][len(preset):]
    res_final = res_1.split("\n")[3:8 if len(res_1.split("\n")) > 8 else -1]
    return res_final


def question_generation(crime, is_victim):
    victim_dict = {'피해': 'victim', '가해': 'perpetrator'}
    file = open(
        f"asset/{victim_dict[is_victim]}_question_generation_prompt.txt", "r")
    question_generation_prompt = ''.join(file.readlines())
    file.close()
    question = ''
    while (len(question) == 0):
        question = make_question(question_generation_prompt, crime)
    return question


def request_clova_requisite(preset, target, requisite):
    input = preset + target + '\n' + requisite + ':'
    request_data = {
        'text': input,
        'maxTokens': 50,
        'temperature': 0.2,
        'topK': 2,
        'topP': 0.8,
        'repeatPenalty': 8.0,
        'start': '',
        'restart': '',
        'stopBefore': ["###"],
        'includeTokens': True,
        'includeAiFilters': True,
        'includeProbs': True
    }
    response = completion_executor.execute(request_data)
    return input, response


def make_requisite_new(preset, target, requisite):
    res = 'Error'
    res_final = 'None'
    while (res == 'Error'):
        res = request_clova_requisite(
            preset, target.strip(), requisite)[1]
        if res != 'Error':
            res_final = res.split(f"{requisite}:")[1].split('\n')[0]
    return res_final


# def make_requisite(preset, target, requisite):
#     res = 'Error'
#     res_final = 'None'
#     while (res == 'Error'):
#         res = request_clova_requisite(preset, target, requisite)[1]
#         if res != 'Error':
#             if "\n##" in res:
#                 res = res.replace("\n##", '')
#             res_final = res.split(f"{requisite}:")[1].split('\n')[0]
#     return res_final
def make_requisite(preset, target, requisite):
    res = 'Error'
    res_final = 'None'
    while (res == 'Error'):
        res = request_clova_requisite(preset, target, requisite)[1]
        if res != 'Error':
            if "###" in res:
                res = res[len(preset):].split("\n###")[0]
            else:
                res = res[len(preset):]
            res_final = res.split(f"{requisite}:")[1]
    return res_final


def make_requisite_prompt(preset, crime, requisite):
    input = preset + f'질문글에서 {crime}죄 관련 {requisite} 부분을 추출합니다. {requisite}이란'
    request_data = {
        'text': input,
        'maxTokens': 50,
        'temperature': 0.2,
        'topK': 5,
        'topP': 0.8,
        'repeatPenalty': 5.0,
        'start': '',
        'restart': '',
        'stopBefore': ["###"],
        'includeTokens': True,
        'includeAiFilters': True,
        'includeProbs': True
    }
    response = completion_executor.execute(request_data)
    return input, response

# condition : target_requisite not in target


def requisite_extraction(question, crime, requisite):
    start_time = time.time()
    requisite_dict = {'모욕성': 'contempt', '특정성': 'target', '공연성': 'public'}
    if requisite in list(requisite_dict.keys()):
        file = open(
            f"asset/requisite_{requisite_dict[requisite]}_prompt.txt", "r")
        requisite_extraction_prompt = ''.join(file.readlines())
        file.close()
        target = preprocessing(question)
        target_requisite = 'None'
        while (target_requisite not in target):
            target_requisite = make_requisite(
                requisite_extraction_prompt, target, requisite).strip()
            print(target_requisite)
            if time.time() - start_time > 30:
                return None
                break
    else:
        file = open(
            f"asset/requisite_all_prompt.txt", "r")
        requisite_extraction_prompt = ''.join(file.readlines())
        file.close()
        requisite_extraction_prompt = make_requisite_prompt(
            requisite_extraction_prompt, crime, requisite)[1]
        target = preprocessing(question)
        target_requisite = 'None'
        while (target_requisite not in target):
            target_requisite = make_requisite_new(
                requisite_extraction_prompt, target, requisite).strip()
            print(target_requisite)
            if time.time() - start_time > 30:
                return None
                break
    return target_requisite
