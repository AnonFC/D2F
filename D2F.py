import json
import time
import string
import re
import os
from IPython.utils import io
import urllib.request
from datetime import datetime
import openai
from serpapi import GoogleSearch
from tqdm import tqdm
from config import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from argparse import ArgumentParser, Namespace
import argparse

openai.api_key = OPENAI_API_KEY
serpapi_key = SERPAPI_KEY

error_log_file = 'logs/mylog.txt'
if not os.path.exists(error_log_file):
    os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
    with open(error_log_file, 'w') as f:
        f.write('')

def my_log(txt):
    with open(error_log_file, 'a') as f:
        f.write(txt + '\n')


def call_gpt(cur_prompt, stop=["\n"]):
    reasoner_messages = [{"role": "user", "content": cur_prompt}]
    print("-------- call gpt --------")
    print(reasoner_messages)
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4o-2024-05-13",
        messages=reasoner_messages,
        # stop=stop
    )
    returned = completion['choices'][0]["message"]["content"]
    print("-------- returned start --------")
    print(returned)
    print("-------- returned end   --------")
    return returned


def serp_search(query):
    params = {
        "api_key": serpapi_key,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    res = None
    with io.capture_output() as captured:  # disables prints from GoogleSearch
        search = GoogleSearch(params)
        res = search.get_dict()
    return res

def is_social_media(link):
    sites = ['mobile.twitter.com',
        'twitter.com',
        'x.com',
        'toolbox.google.com',
        'reddit.com',
        'snopes.com',
        'facebook.com',
        'instagram.com',
        'linkedin.com',
        'pinterest.com',
        'snapchat.com',
        'tumblr.com',
        'tiktok.com',
        'youtube.com',
        'vimeo.com',
        'whatsapp.com',
        'quora.com',
        'weibo.com',
        'yelp.com',
        'sina.com.cn',
        'snopes.com',
        'politifact.com',
        'www.truthorfiction.com',
        'www.factcheck.org',
        'fullfact.org',
        'apnews.com',
        'uk.reuters.com',
    ]
    for site in sites:
        if site in link:
            return True
    return False

def get_answer(question):
    params = {
        "api_key": serpapi_key,
        "engine": "google",
        "q": question,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    with io.capture_output() as captured:  # disables prints from GoogleSearch
        search = GoogleSearch(params)
        res = search.get_dict()

    # google 搜索 evidence。简单过滤掉 fact-checking 的网址。
    answers = []
    if "organic_results" in res.keys():
        for idx in range(len(res["organic_results"])):
            if 'snippet' in res["organic_results"][idx].keys():
                if is_social_media(res["organic_results"][idx]['link']) :
                    continue
                toret = res["organic_results"][idx]['snippet']
                answers.append(toret)
            if (idx + 1) == len(res["organic_results"]):
                toret = None
    else:
        toret = None
    # return toret
    return answers[:5]


def disambiguate(claim):
    prompt = """AMBIGUITY CRITERIA: Ambiguity manifests in diverse forms, including: 
- Similar names denoting distinct entities. 
- Varied interpretations stemming from insufficient information. 
- Multiple understandings arising from vague or unclear information. 

Instructions: 
- Identify the main SUBJECT within the claim. 
- Determine if the SUBJECT is ambiguous according to the provided AMBIGUITY CRITERIA. 
- Utilize your world knowledge to enumerate potential DISAMBIGUATIONS for the identified SUBJECT. 
- Strictly adhere to the context information provided in the claim and avoid introducing DISAMBIGUATION options that are inconsistent with the claim's content. 
- Specify the TYPE of information employed for disambiguation based on the list of DISAMBIGUATIONS. 
- If the SUBJECT does not have ambiguous interpretations, return None 
- Provide an explanation of the method used to arrive at the final response. 

Format your response as a combination of explanation and a dictionary with the following structure: 
##EXPLANATION##: 
<step-by-step-explanations> 
##RESPONSE##: 
{"subject": <subject>, "disambiguations":[ <instance-1>, <instance-2>..], "disambiguation_type": <type>} 

Example 1: 
##CLAIM##: David Heyman, born in 1961 in England, is the founder of Heyday Films. 
##EXPLANATION##: The SUBJECT of the claim is "David Heyman". Based on my world knowledge, there are multiple individuals who share similar names, such as "David Heyman - the British film producer" and "David Heyman - the Chairman of the Board of UK HPA."  To differentiate between them, it is crucial to consider their respective occupations. This criterion offers a  clearer disambiguation compared to nationality, as both individuals are British and thus nationality alone does not  provide sufficient distinguishing information. 
##RESPONSE##: 
{"subject": "David Heyman", "disambiguations": ["David Heyman - British film producer, founder of Heyday Films", "David L. Heyman - Chairman of the Board of UK HPA"], "disambiguation_type": "Occupation"} 

Example 2: 
##CLAIM##: Ruth Bader Ginsburg served as a Supreme Court justice. 
##EXPLANATION##: The SUBJECT is "Ruth Bader Ginsburg". According to my world knowledge, this is a unique individual and I am not aware  of any other individuals/entities with a similar name. Hence, there are no ambiguous interpretations of this SUBJECT and the claim requires no further disambiguation. 
##RESPONSE##: 
{"subject": "Ruth Bader Ginsburg", "disambiguations": "None"} 

Example 3: 
##CLAIM##: Charles Osgood, the american television commentator, is best known for hosting CBS News Sunday Morning. 
##EXPLANATION##: The SUBJECT in focus is "Charles Osgood". Based on my world knowledge, there are two notable individuals with similar names: "Charles Osgood - American radio and television commentator" and "Charles E. Osgood - American psychologist." Given the ambiguity surrounding the name,  specifying the individual's profession serves as an apt disambiguation method. 
##RESPONSE##: 
{"subject": "Charles Osgood", "disambiguations": ["Charles Osgood - American radio and television commentator",  "Charles E. Osgood - American psychologist"], "disambiguation_type": "Profession"} 

Similarly, disambiguate the following claim by detecting the main SUBJECT and disambiguation information for the SUBJECT using your world knowledge.  Generate an EXPLANATION followed by dictionary-formatted RESPONSE. 
##CLAIM##: [claim] 
##EXPLANATION##:""".replace("[claim]", claim)
    return call_gpt(prompt)



def question_answering(claim, subclaims, question, evidence, variable):
    prompt = ("""Assign a value to the variable {answer_1} based on the content in the evidence. Ensure that each subclaim forms a complete and coherent sentence. Only output the value. If no answer can be found in the evidence, output 'unknown'.

##SUBCLAIMS##: 
[subclaims]
##QUESTION##: 
[question]
##EVIDENCE##: 
[evidence]
##ANSWER##:
[variable] =""".replace("[claim]", claim)
              .replace('[subclaims]', "\n".join(subclaims))
              .replace("[question]", question)
              .replace('[evidence]', evidence)
              .replace("[variable]", variable))
    return call_gpt(prompt)

def decompose(claim):
    prompt = """Decompose the following claim.

DECOMPOSITION CRITERIA: 
- Each subclaim should be a single, verifiable fact. Avoid compound statements.
- Subclaims should be independently verifiable, meaning the truth of one subclaim does not depend on another.
- Subclaims should be directly related to the main claim and contribute to its overall verification.
- Subclaims should be specific and detailed enough to allow for precise fact-checking.
- Subclaims should be clear and unambiguous to avoid misinterpretation.
- Ensure that the decomposition is logical and consistent with the original claim.
- Utilize your internal knowledge. If you lack some knowledge, replace it with a variable.
- Generate corresponding questions to verify the corresponding questions. The questions should be easy to retrieve from knowledge base.

Example 1:
##CLAIM##: The Rodney King riots took place in a county in the U.S. with a population of over one million.
##SUBCLAIMS##:
The Rodney King riots took place in {answer_1}.
The population of {answer_1} is {answer_2}.
{answer_2} is over one million.
##QUESTIONS##:
Where did The Rodney King riots take place? {answer_1}
What is the population of {answer_1}? {answer_2}

Example 2:
##CLAIM##: Says 21,000 Wisconsin residents got jobs in 2011, but 18,000 of them were in other states.
##SUBCLAIMS##:
21,000 Wisconsin residents got jobs in 2011.
18,000 Wisconsin residents who got jobs in 2011 were employed in other states.
##QUESTIONS##:
How many Wisconsin residents got jobs in 2011? {answer_1}
How many Wisconsin residents who got jobs in 2011 were employed in other states? {answer_2}


##CLAIM##: """ + claim
    return call_gpt(prompt)

def extract_questions(response):
    """Extract questions from decompose(claim) response."""
    lines = response.splitlines()
    claim = ''
    subclaims = []
    questions = []
    flag_subclaim = False
    flag_questions = False
    for line in lines:
        if line.strip() == '':
            continue
        if line.startswith('##CLAIM##:'):
            claim = line.split('##CLAIM##:')[1]
        if line.startswith('##SUBCLAIMS##:'):
            flag_subclaim = True
            flag_questions = False
            continue
        if line.startswith('##QUESTIONS##:'):
            flag_subclaim = False
            flag_questions = True
            continue
        if flag_subclaim:
            subclaims.append(line)
        if flag_questions:
            questions.append(line)
    return claim, subclaims, questions


def factchecking_LIAR_RAW(claim, responses):
    questions = []
    answers = []
    for q in responses['questions']:
        questions.append(q['question'])
        qq = q['question'].split("{answer_")[0]
        print(q['subject'], "\n", qq)
        ans = get_answer(qq)
        answers = answers + ans
    answers_str = "\n".join(['- ' + a for a in answers])
    questions_str = "\n".join(questions)
    subclaims_str = "\n".join(responses['subclaims'])
    prompt = ("""Verify the following claim.

Use the Truth-O-Meter to reflect the relative accuracy of the claim. The meter has six ratings, in decreasing level of truthfulness:
- TRUE : The claim is accurate and there’s nothing significant missing.
- MOSTLY TRUE : The claim is accurate but needs clarification or additional information.
- HALF TRUE : The claim is partially accurate but leaves out important details or takes things out of context.
- MOSTLY FALSE : The claim contains an element of truth but ignores critical facts that would give a different impression.
- FALSE : The claim is not accurate.
- PANTS ON FIRE : The claim is not accurate and makes a ridiculous claim. 

Instructions: 
- Replace the variables in the subclaim with the correct answer based on what is in the EVIDENCE.
- Fact-check each suclaim and label it as a label in the Truth-O-Meter.
- Based on the validation results of each subclaim, give the final result for this claim.

Format your response as a dictionary with the following structure: 
{"subclaim_ratings": [{"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}, {"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}], "final_rating": <final rating>, "explanation": <overall explanation>}

Example:
##CLAIM##: The Rodney King riots took place in the most populous county in the USA.
##QUESTIONS##:
Where did The Rodney King riots take place? {answer_1}
##SUBCLAIMS##:
The Rodney King riots took place in {answer_1}.
{answer_1} is the most populous county in the USA.
##EVIDENCE##:
- The riots first began at an intersection in South Los Angeles — Florence and Normandie — according to news reports and firsthand accounts in the ...
- Emotions were still running high more than a year later during the trial of the officers conducted in Simi Valley, a suburb of Los Angeles. On ...
- Lake View Terrace: Rodney King beating ... On March 3, 1991, police arrested Rodney King at the intersection of Foothill Boulevard and Osborne ...
- The officers' trial is moved to Simi Valley, a nearly all-white suburb 30 miles north of downtown Los Angeles that is home to a large number of L.A. police ...
- May 1, 1992, 3 p.m.: West of Koreatown, in Beverly Hills, Rodney King appeared outside his lawyer's office for a news conference. Quiet and ...
##RESPONSE##: 
{"subclaim_ratings": [{"subclaim": "The Rodney King riots took place in Los Angeles.", "rating": "TRUE", "explanation": "The evidence consistently indicates that the Rodney King riots took place in Los Angeles."}, {"subclaim": "Los Angeles is the most populous county in the USA.", "rating": "TRUE", "explanation": "Los Angeles County is indeed the most populous county in the United States."}], "final_rating": "TRUE", "explanation": "Given that both subclaims are rated as TRUE, the overall claim is also rated as: TRUE"}

##CLAIM##: [claim]
##QUESTIONS##:
[questions]
##SUBCLAIMS##:
[subclaims]
##EVIDENCE##:
[evidence]
##RESPONSE##:""".replace("[claim]", claim)
              .replace("[questions]", questions_str)
              .replace("[subclaims]", subclaims_str).
              replace("[evidence]", answers_str))
    return call_gpt(prompt)


def factchecking_RAWFC(claim, questions, subclaims, evidence_list):
    answers_str = "\n".join(['- ' + a for a in evidence_list])
    questions_str = "\n".join(questions)
    subclaims_str = "\n".join(subclaims)
    prompt = ("""Verify the following claim.

Use the following meter to reflect the relative accuracy of the claim. The meter has three ratings, in decreasing level of truthfulness:
- TRUE: This rating indicates that the primary elements of a claim are demonstrably true.
- FALSE: This rating indicates that the primary elements of a claim are demonstrably false.
- MIXTURE: This rating indicates that a claim has significant elements of both truth and falsity to it such that it could not fairly be described by any other rating.

Instructions: 
- Replace the variables in the subclaim with the correct answer based on what is in the EVIDENCE.
- Fact-check each suclaim and label it as a label in the above meter.
- Based on the evidence and your internal knowledge, give the final result for this claim.

Format your response as a dictionary with the following structure: 
{"subclaim_ratings": [{"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}, {"subclaim": <subclaim-1>, "rating": <rating-1>, "explanation": <explanation-1>}], "final_rating": <final rating>, "explanation": <overall explanation>}

Example:
##CLAIM##: The Rodney King riots took place in the most populous county in the USA.
##QUESTIONS##:
Where did The Rodney King riots take place? {answer_1}
##SUBCLAIMS##:
The Rodney King riots took place in {answer_1}.
{answer_1} is the most populous county in the USA.
##EVIDENCE##:
- The riots first began at an intersection in South Los Angeles — Florence and Normandie — according to news reports and firsthand accounts in the ...
- List ; 1, Los Angeles · California, 10,509.87, 4,057.88, 10,014,009 ; 2, Cook · Illinois, 2,448.38, 945.33, 5,275,541 ; 3, Harris · Texas, 4,411.99, 1,703.48 ...
##RESPONSE##: 
{"subclaim_ratings": [{"subclaim": "The Rodney King riots took place in Los Angeles.", "rating": "TRUE", "explanation": "The evidence consistently indicates that the Rodney King riots took place in Los Angeles."}, {"subclaim": "Los Angeles is the most populous county in the USA.", "rating": "TRUE", "explanation": "Los Angeles County is indeed the most populous county in the United States."}], "final_rating": "TRUE", "explanation": "Given that both subclaims are rated as TRUE, the overall claim is also rated as: TRUE"}

##CLAIM##: [claim]
##QUESTIONS##:
[questions]
##SUBCLAIMS##:
[subclaims]
##EVIDENCE##:
[evidence]
##RESPONSE##:""".replace("[claim]", claim)
              .replace("[questions]", questions_str)
              .replace("[subclaims]", subclaims_str).
              replace("[evidence]", answers_str))
    return call_gpt(prompt)


def read_res(res_file):
    with open(res_file, 'r') as f:
        text = f.read()
    arr = text.split("---------------------")
    json_str = arr[3]
    if json_str.find('```json') > -1:
        json_str = json_str.replace("```json", "").replace('```', '')
    pred = json.loads(json_str)['final_rating']
    return pred


def num_label(label):
    l = label.strip().lower()
    d = {
        'true': 0,
        'TRUE': 0,

        'half-true': 1,
        'HALF TRUE': 1,
        'MIXTRUE': 1,
        'mixture': 1,
        'half': 1,
        'half true': 1,

        'false': 2,
        'FALSE': 2,

        'mostly-true': 3,
        'MOSTLY TRUE': 3,
        'mostly true': 3,

        'barely-true': 4,
        'MOSTLY FALSE': 4,
        'mostly false': 4,

        'pants-fire': 5,
        'pants on fire': 5,
        'PANTS ON FIRE': 5
    }
    return d[l]


def test_LIAR_RAW(now_time):
    dataset_path = './data/LIAR-RAW/test.json'

    log_path = f'./log/{now_time}/LIAR-RAW/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    my_log('=== start ' + log_path)

    with open(dataset_path, 'r') as json_file:
        json_list = json.load(json_file)

    for item in tqdm(json_list):
        label = item["label"]
        claim = item["claim"]
        idx = item["event_id"]

        to_file = os.path.join(log_path, str(idx) + '.txt')
        if os.path.exists(to_file):
            continue

        try:
            print('---------- idx ----------')
            print(str(idx) + '\n')

            # claim = "Georgia has had ʺmore bank failures than any other state.ʺ"
            res1 = decompose(claim)
            responses = res1.split("##RESPONSE##:")[-1].strip()
            responses = json.loads(responses)
            res2 = factchecking_LIAR_RAW(claim, responses)

            with open(to_file, 'w') as f:
                f.write(res1)
                f.write('---------------------\n')
                f.write(res2)
                f.write('---------------------\n')
                f.write("ground truth: {}".format(label))
            print("saved to", to_file)
        except Exception as e:
            my_log("-- error: {}, idx: {}".format(str(e), idx))


def eval_LIAR_RAW(runtime=''):
    dataset_name = 'LIAR-RAW'
    dataset_path = './data/{}/test.json'.format(dataset_name)
    if runtime == '':
        runtime = '2024-08-02-22-36'
    results_path = 'log/{}/{}/'.format(runtime, dataset_name)

    with open(dataset_path, 'r') as json_file:
        json_list = json.load(json_file)

    y_pred = []
    y_true = []
    for result in json_list:
        label = result["label"]
        claim = result["claim"]
        idx = result["event_id"]

        res_file = os.path.join(results_path, str(idx) + '.txt')
        if not os.path.exists(res_file):
            continue
        print(res_file)
        pred = read_res(res_file)
        if pred is None:
            print("Error: the pred is None")
            exit()
        print(claim, '\n', label, pred, '\n')
        y_true.append(num_label(label))
        y_pred.append(num_label(pred))

    target_names = ['true', 'half-true', 'false', 'mostly-true', 'barely-true', 'pants-fire']
    report_txt = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=False,
                                       zero_division=1)
    print(report_txt)


def wikidata_search(query):
    # 如果缓存中没有结果，调用API进行搜索
    service_url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': query,
        'language': 'en',
        'limit': 20,
        'format': 'json'
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    try:
        if len(response['search']) > 0:
            qid = response['search'][0]['id']
            desc = response['search'][0]['description']
            label = response['search'][0]['label']
            item = {
                'qid': qid,
                'label': label,
                'desc': desc
            }
            return item
    except Exception as e:
        print(response)
        print(str(e))
    return None

def test_RAWFC(now_time):
    dataset_path = './data/RAWFC/test'

    log_path = f'./log/{now_time}/RAWFC/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    my_log('=== start ' + log_path)

    filelist = [f for f in os.listdir(dataset_path)]
    for ff in tqdm(filelist):
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            to_file = os.path.join(log_path, str(idx) + '.txt')
            if os.path.exists(to_file):
                continue

            try:
                print('---------- idx ----------')
                print(str(idx) + '\n')

                # claim = "Georgia has had ʺmore bank failures than any other state.ʺ"
                resp = decompose(claim)
                _, subclaims, questions = extract_questions(resp)

                # QA
                answers = {}  # e.g.  {'{answer_1}': 'Finland'}
                evidence_list = []
                for q in questions:
                    print("question:", q)
                    query = q
                    flag_searched = False

                    # 正则表达式提取变量
                    pattern = r'\{[^}]*\}'
                    matches = re.findall(pattern, q)

                    for m in matches:
                        # 如果问题中存在已知答案，把它替换
                        if m in answers.keys():
                            query = q.replace(m, answers[m])
                            continue

                        # 如果存在未知变量，搜索答案
                        query = q.replace(m, "")
                        ans = 'unknown'

                        # Google 搜索答案。排名考前的搜索结果优先，一旦找到答案就不再管后面的evidence。
                        flag_searched = True
                        search_res = get_answer(query)
                        for evidence in search_res:
                            value = question_answering(claim, subclaims, q, evidence, m)
                            print("value:", value)
                            if value != 'unknown':
                                ans = value
                                evidence_list.append(evidence)
                                break
                        answers[m] = ans
                        if ans == 'unknown':
                            evidence_list.append(search_res[0])

                    # 如果没被搜索过，说明这是一个不含变量的问题
                    # 以第一个搜索结果为主
                    if not flag_searched:
                        search_res = get_answer(query)
                        evidence_list.append(search_res[0])

                print("------- QA -------")
                print("answers:")
                for k, v in answers.items():
                    print(k, v)
                print("evidence:")
                for e in evidence_list:
                    print(e)

                resp2 = factchecking_RAWFC(claim, questions, subclaims, evidence_list)
                with open(to_file, 'w') as f:
                    f.write(resp)
                    f.write('---------------------\n')
                    f.write(json.dumps(answers))
                    f.write('---------------------\n')
                    f.write("\n".join(evidence_list))
                    f.write('---------------------\n')
                    f.write(resp2)
                    f.write('---------------------\n')
                    f.write("ground truth: {}".format(label))
                print("saved to", to_file)
            except Exception as e:
                my_log("-- error: {}, idx: {}".format(str(e), idx))


def eval_RAWFC(runtime=''):
    dataset_name = 'RAWFC'
    dataset_path = './data/{}/test'.format(dataset_name)

    results_path = 'log/{}/{}/'.format(runtime, dataset_name)

    y_pred = []
    y_true = []
    filelist = [f for f in os.listdir(dataset_path)]
    for ff in tqdm(filelist):
        full_path = os.path.join(dataset_path, ff)
        if os.path.isdir(full_path):
            continue
        if ff.endswith('.json'):
            with open(full_path, 'r') as json_file:
                item = json.load(json_file)

            label = item["label"]
            claim = item["claim"]
            idx = item["event_id"]

            res_file = os.path.join(results_path, str(idx) + '.txt')
            if not os.path.exists(res_file):
                continue
            print(res_file)
            pred = read_res(res_file)
            if pred is None:
                print("Error: the pred is None")
                exit()
            print(claim, '\n', label, pred, '\n')
            y_true.append(num_label(label))
            y_pred.append(num_label(pred))

    target_names = ['true', 'mixtrue', 'false']
    print(y_pred)
    print(y_true)
    report_txt = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=False,
                                       zero_division=1)
    print(report_txt)


def test_one_RAWFC(claim):
    resp = decompose(claim)
    _, subclaims, questions = extract_questions(resp)

    # QA
    answers = {}  # e.g.  {'{answer_1}': 'Finland'}
    evidence_list = []
    for q in questions:
        print("question:", q)
        query = q
        flag_searched = False

        # 正则表达式提取变量
        pattern = r'\{[^}]*\}'
        matches = re.findall(pattern, q)

        for m in matches:
            # 如果问题中存在已知答案，把它替换
            if m in answers.keys():
                query = q.replace(m, answers[m])
                continue

            # 如果存在未知变量，搜索答案
            query = q.replace(m, "")
            ans = 'unknown'

            # Google 搜索答案。排名考前的搜索结果优先，一旦找到答案就不再管后面的evidence。
            flag_searched = True
            search_res = get_answer(query)
            for evidence in search_res:
                value = question_answering(claim, subclaims, q, evidence, m)
                print("value:", value)
                if value != 'unknown':
                    ans = value
                    evidence_list.append(evidence)
                    break
            answers[m] = ans
            if ans == 'unknown':
                evidence_list.append(search_res[0])

        # 如果没被搜索过，说明这是一个不含变量的问题
        # 以第一个搜索结果为主
        if not flag_searched:
            search_res = get_answer(query)
            evidence_list.append(search_res[0])

    print("------- QA -------")
    print("answers:")
    for k, v in answers.items():
        print(k, v)
    print("evidence:")
    for e in evidence_list:
        print(e)

    resp2 = factchecking_RAWFC(claim, questions, subclaims, evidence_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="D2F",
        add_help=True,
    )
    parser.add_argument(
        "--now_time",
        default="",
        type=str,
        required=False,
        help="now_time"
    )
    parser.add_argument(
        "--data_set",
        default="RAWFC",
        type=str,
        choices=["LIAR_RAW", "RAWFC"],
        required=False,
        help="data_set: [LIAR_RAW, RAWFC]"
    )
    hparams = parser.parse_args()
    if hparams.now_time == "":
        now_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    else:
        now_time = hparams.now_time

    if hparams.data_set == 'LIAR_RAW':
        test_LIAR_RAW(now_time)
        eval_LIAR_RAW(now_time)
    else:
        test_RAWFC(now_time)
        eval_RAWFC(now_time)


